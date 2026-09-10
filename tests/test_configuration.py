import copy
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from pydantic import ValidationError

from autoresttest import autoresttest as app
from autoresttest.agents import ValueAgent
from autoresttest.config import apply_config_overrides, load_config
from autoresttest.graph import OperationGraph, RequestGenerator
from autoresttest.llm import LanguageModel, SmartValueGenerator
from autoresttest.marl import QLearning
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    RequestData,
    SchemaProperties,
)
from autoresttest.specification import SpecificationParser
from autoresttest.utils import attempt_fix_json, get_combinations


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.base = load_config(self.root / "configurations.toml.example")
        self.config = apply_config_overrides(
            {
                "llm": {
                    "engine": "test-model",
                    "api_base": "http://provider.invalid/v1",
                    "creative_temperature": 0.25,
                    "strict_temperature": 0,
                    "max_tokens": 100,
                    "timeout_seconds": 9,
                },
                "api": {"request_timeout_seconds": 4},
                "custom_headers": {"X-Test": "configured"},
                "agent": {"max_total_combinations": 3, "value": {"max_workers": 2}},
            },
            self.base,
        )
        self.operation = OperationProperties(
            operation_id="probe",
            endpoint_path="/items",
            http_method="get",
            parameters={
                ("id", "query"): ParameterProperties(
                    name="id",
                    in_value="query",
                    schema=SchemaProperties(type="integer"),
                )
            },
        )

    def make_graph(self):
        graph = OperationGraph(
            "unused", "probe", SimpleNamespace(config=self.config), Mock()
        )
        graph.add_operation_node(self.operation)
        graph.assign_request_generator(
            RequestGenerator(graph, "http://target.invalid", is_naive=False)
        )
        return graph

    def run_cli(self, arguments, wizard_overrides=None):
        with (
            patch.object(sys, "argv", ["autoresttest", *arguments]),
            patch.object(app, "load_config", return_value=self.base) as load,
            patch.object(app, "TUIDisplay") as tui,
            patch.object(app, "ConfigWizard") as wizard,
            patch.object(app, "AutoRestTest") as runner,
        ):
            wizard.return_value.run.return_value = wizard_overrides or {}
            app.main()
        load.assert_called_once_with()
        effective = runner.call_args.kwargs["config"]
        summary = tui.return_value.print_config_summary.call_args.args[0]
        self.assertEqual(summary["LLM Engine"], effective.llm.engine)
        self.assertEqual(summary["Specification"], effective.spec.location)
        self.assertEqual(
            summary["Duration"], f"{effective.request_generation.time_duration}s"
        )
        return effective, wizard

    def test_cli_overrides_preserve_unrelated_wizard_settings(self):
        effective, wizard = self.run_cli(
            ["--spec", "cli.yaml", "--time", "20"],
            {
                "llm": {
                    "engine": "wizard-model",
                    "api_base": "http://wizard.invalid/v1",
                },
                "spec": {"location": "wizard.yaml"},
                "request_generation": {"time_duration": 10, "mutation_rate": 0},
            },
        )
        self.assertEqual(effective.spec.location, "cli.yaml")
        self.assertEqual(effective.request_generation.time_duration, 20)
        self.assertEqual(effective.request_generation.mutation_rate, 0)
        self.assertEqual(effective.llm.engine, "wizard-model")
        self.assertEqual(effective.llm.api_base, "http://wizard.invalid/v1")
        self.assertIs(wizard.call_args.kwargs["config"], self.base)

    def test_optional_wizard_and_cli_preserve_file_defaults(self):
        effective, wizard = self.run_cli(["--skip-wizard"])
        self.assertEqual(effective, self.base)
        wizard.assert_not_called()
        effective, _ = self.run_cli([], {"llm": {"engine": "wizard-model"}})
        self.assertEqual(effective.llm.engine, "wizard-model")
        self.assertEqual(effective.request_generation, self.base.request_generation)
        effective, _ = self.run_cli(["--skip-wizard", "--time", "20"])
        self.assertEqual(effective.request_generation.time_duration, 20)
        self.assertEqual(effective.llm, self.base.llm)

    def test_nested_overrides_preserve_false_zero_and_inputs(self):
        before = self.config.model_dump()
        overrides = {
            "agent": {"value": {"parallelize": False}},
            "llm": {"creative_temperature": 0},
        }
        original = copy.deepcopy(overrides)
        result = apply_config_overrides(overrides, self.config)
        self.assertFalse(result.agent.value.parallelize)
        self.assertEqual(result.agent.value.max_workers, 2)
        self.assertEqual(result.llm.creative_temperature, 0)
        self.assertEqual(self.config.model_dump(), before)
        self.assertEqual(overrides, original)
        with self.assertRaises(ValidationError):
            apply_config_overrides(
                {"request_generation": {"time_duration": 0}}, self.config
            )

    def test_invalid_explicit_cli_duration_is_validated(self):
        with self.assertRaises(ValidationError):
            self.run_cli(["--skip-wizard", "--time", "0"])

    def test_llm_uses_each_runs_config_and_does_not_share_provider_cache(self):
        other = apply_config_overrides(
            {"llm": {"api_base": "http://other.invalid/v1"}}, self.config
        )
        with (
            patch.dict("os.environ", {"API_KEY": "test-key"}),
            patch("autoresttest.llm.llm.OpenAI") as client,
            patch.object(LanguageModel, "cache", {}),
        ):
            client.return_value.chat.completions.create.side_effect = [
                SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(message=SimpleNamespace(content=value))],
                )
                for value in ("first provider", "second provider")
            ]
            first = LanguageModel(config=self.config)
            second = LanguageModel(config=other)
            self.assertEqual(first.query("same prompt"), "first provider")
            self.assertEqual(second.query("same prompt"), "second provider")
            self.assertEqual(first.query("same prompt"), "first provider")
            self.assertEqual(client.return_value.chat.completions.create.call_count, 2)
            self.assertEqual(
                client.call_args_list[0].kwargs["base_url"], self.config.llm.api_base
            )
            self.assertEqual(
                client.call_args_list[1].kwargs["base_url"], other.llm.api_base
            )
            self.assertEqual(client.call_args.kwargs["timeout"], 9)
            request = client.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(
                (request["model"], request["temperature"], request["max_tokens"]),
                ("test-model", 0.25, 100),
            )

    def test_parser_uses_configured_validation_and_recursion(self):
        config = apply_config_overrides(
            {"spec": {"strict_validation": False, "recursion_limit": 2}}, self.config
        )
        with (
            patch(
                "autoresttest.specification.specification_parser.ResolvingParser"
            ) as strict,
            patch(
                "autoresttest.specification.specification_parser.LenientResolvingParser"
            ) as lenient,
        ):
            parser = SpecificationParser("unused", config=config)
            strict.assert_not_called()
            self.assertEqual(lenient.call_args.kwargs["recursion_limit"], 2)
            self.assertIs(parser.config, config)

    def test_both_api_request_paths_use_run_headers_and_timeout(self):
        graph = self.make_graph()
        learner = QLearning(graph)
        response = requests.Response()
        response.status_code, response._content = 200, b"{}"
        with patch("requests.get", return_value=response) as send:
            learner.send_operation(self.operation, {}, None, None)
            graph.request_generator.send_operation_request(
                RequestData(
                    endpoint_path="/items",
                    http_method="get",
                    parameters={},
                    request_body=None,
                    operation_properties=self.operation,
                )
            )
        self.assertEqual(send.call_count, 2)
        for call in send.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 4)
            self.assertEqual(call.kwargs["headers"]["X-Test"], "configured")

    def test_runner_passes_config_into_graph_and_request_generator(self):
        with (
            patch.object(app, "construct_db_dir"),
            patch.object(app, "SpecificationParser") as parser,
        ):
            parser.return_value.config = self.config
            parser.return_value.get_api_url.return_value = "http://target.invalid"
            runner = app.AutoRestTest("unused", self.config, Mock())
            graph = runner.init_graph("probe", "unused", Mock())
        self.assertIs(parser.call_args.kwargs["config"], self.config)
        self.assertIs(graph.config, self.config)
        self.assertIs(graph.request_generator.config, self.config)

    def test_request_value_generation_receives_run_config(self):
        graph = self.make_graph()
        with patch(
            "autoresttest.graph.request_generator.SmartValueGenerator"
        ) as generator:
            generator.return_value.generate_parameters.return_value = {
                ("id", "query"): 42
            }
            generator.return_value.generate_request_body.return_value = None
            request = graph.request_generator.make_request_data(self.operation)
        self.assertIs(generator.call_args.kwargs["config"], self.config)
        self.assertEqual(request.parameters, {("id", "query"): 42})

    def test_value_generation_and_json_repair_receive_run_config(self):
        with patch("autoresttest.llm.value_generator.LanguageModel") as model:
            generator = SmartValueGenerator(self.operation, config=self.config)
            self.assertIs(model.call_args.kwargs["config"], self.config)
            model.return_value.query.return_value = "invalid JSON"
            with patch(
                "autoresttest.llm.value_generator.attempt_fix_json", return_value={}
            ) as repair:
                generator.generate_parameters()
                self.assertIs(repair.call_args.kwargs["config"], self.config)
        with patch("autoresttest.llm.LanguageModel") as model:
            model.return_value.query.return_value = "{}"
            self.assertEqual(attempt_fix_json("broken", config=self.config), {})
            self.assertIs(model.call_args.kwargs["config"], self.config)
            self.assertEqual(model.call_args.kwargs["temperature"], 0)

    def test_parallel_value_workers_receive_run_config(self):
        graph = self.make_graph()
        agent = ValueAgent(graph)
        with (
            patch.object(
                graph.request_generator, "create_and_send_request", return_value=None
            ),
            patch(
                "autoresttest.graph.request_generator.SmartValueGenerator"
            ) as generator,
        ):
            generator.return_value.generate_value_agent_params.return_value = {
                ("id", "query"): [42]
            }
            generator.return_value.generate_value_agent_body.return_value = {}
            agent.initialize_q_table()
        self.assertIs(generator.call_args.kwargs["config"], self.config)
        self.assertEqual(agent.q_table["probe"]["params"][("id", "query")], [[42, 0]])

    def test_value_agent_uses_worker_count_and_can_disable_parallelism(self):
        graph = self.make_graph()
        agent = ValueAgent(graph)
        with patch.object(
            graph.request_generator, "generate_value_tables_parallel"
        ) as parallel:
            agent.initialize_q_table()
            self.assertEqual(parallel.call_args.kwargs["max_workers"], 2)
        graph.config = apply_config_overrides(
            {"agent": {"value": {"parallelize": False}}}, self.config
        )
        with (
            patch.object(
                graph.request_generator, "generate_value_tables_parallel"
            ) as parallel,
            patch.object(
                graph.request_generator, "value_depth_traversal"
            ) as sequential,
        ):
            agent.initialize_q_table()
            parallel.assert_not_called()
            sequential.assert_called_once()

    def test_combination_limits_follow_the_run(self):
        self.assertLessEqual(len(get_combinations(range(12), config=self.config)), 3)

    def test_import_and_help_do_not_require_a_config_file(self):
        script = (
            "from pathlib import Path\n"
            "from autoresttest.config import config\n"
            "config.CONFIG_PATH = Path('/nonexistent/configurations.toml')\n"
            "import autoresttest.autoresttest as app\n"
            "import sys\n"
            "sys.argv = ['autoresttest', '--help']\n"
            "app.main()\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--skip-wizard", result.stdout)
