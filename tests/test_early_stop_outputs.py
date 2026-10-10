import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from autoresttest import autoresttest as app
from autoresttest.config import apply_config_overrides, load_config
from autoresttest.llm import LanguageModel
from autoresttest.models import (
    OperationProperties,
    ResponseProperties,
    SchemaProperties,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner, run_once  # noqa: E402

CONFIG = apply_config_overrides(
    {"cache": {"use_cached_table": False}},
    load_config(Path(__file__).resolve().parents[1] / "configurations.toml.example"),
)
OPERATION = OperationProperties(
    operation_id="listItems",
    endpoint_path="/items",
    http_method="get",
    responses={
        "200": ResponseProperties(
            status_code="200",
            content={
                "application/json": SchemaProperties(
                    type="object", properties={"id": SchemaProperties(type="integer")}
                )
            },
        )
    },
)
OUTPUT_FILES = {
    "q_tables.json",
    "successful_parameters.json",
    "successful_bodies.json",
    "successful_responses.json",
    "successful_primitives.json",
    "server_errors.json",
    "operation_status_codes.json",
    "report.json",
}


class EarlyStopOutputTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.object(LanguageModel, "failures", Counter()),
            patch.object(LanguageModel, "successful_queries", 0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.output_dir = self.directory / "data" / "probe"
        self.learner = make_learner(OPERATION)
        self.learner.value_agent.initialize_q_table = Mock()
        self.learner.operation_graph.spec_parser = Mock(
            get_api_title=Mock(return_value="Items API")
        )
        self.tui = Mock()

    def patches(self):
        return (
            patch.object(app, "DATA_ROOT", self.directory / "data"),
            patch.object(app, "construct_db_dir"),
            patch.object(app, "QLearning", return_value=self.learner),
            patch.object(
                app, "get_q_table_cache_path", return_value=self.directory / "q_tables"
            ),
            patch.object(app, "InitializationProgressDisplay", MagicMock()),
        )

    def perform_q_learning(self, run):
        """Run setup and testing, where QLearning.run is replaced by `run`."""
        self.learner.run = run
        with ExitStack() as stack:
            for patcher in self.patches():
                stack.enter_context(patcher)
            return app.AutoRestTest("unused", CONFIG, self.tui).perform_q_learning(
                self.learner.operation_graph, "probe"
            )

    def messages(self):
        return [call.args[0] for call in self.tui.print_step.call_args_list]

    def test_outputs_are_written_when_testing_is_interrupted(self):
        def interrupted_run():
            # Record a response, a primitive list and a server error, then stop.
            for status, content in ((200, b'{"id": 7}'), (200, b"[1, 2]"), (500, b"")):
                run_once(
                    self.learner,
                    "listItems",
                    req_params=None,
                    data_source="DEFAULT",
                    status_code=status,
                    content=content,
                )
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.perform_q_learning(interrupted_run)

        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()}, OUTPUT_FILES
        )
        saved = {
            name: json.loads((self.output_dir / name).read_text())
            for name in OUTPUT_FILES
        }
        self.assertEqual(saved["report.json"]["Total Requests Sent"], 3)
        self.assertEqual(saved["successful_responses.json"]["listItems"]["id"], [7])
        self.assertEqual(saved["successful_primitives.json"]["listItems"], [1, 2])
        self.assertEqual(len(saved["server_errors.json"]["listItems"]), 1)
        self.assertEqual(
            self.messages()[-2:],
            [
                "Testing stopped early; saving results so far to data/probe/...",
                "Results saved to: data/probe/",
            ],
        )

    def test_outputs_are_written_when_testing_fails(self):
        with self.assertRaises(RuntimeError):
            self.perform_q_learning(Mock(side_effect=RuntimeError("boom")))
        self.assertTrue((self.output_dir / "report.json").is_file())

    def test_a_failed_save_keeps_the_original_exception_and_previous_file(self):
        def interrupted_run():
            # A value json.dump cannot serialize makes saving fail part-way.
            self.learner.successful_bodies["listItems"] = {"blob": [object()]}
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.perform_q_learning(interrupted_run)

        # q_tables.json from setup was replaced in full; the failed file was not left half-written.
        json.loads((self.output_dir / "q_tables.json").read_text())
        self.assertFalse((self.output_dir / "successful_bodies.json").exists())
        self.assertEqual(list(self.output_dir.glob("*.tmp")), [])
        self.assertTrue(
            any(m.startswith("Could not save results:") for m in self.messages())
        )

    def test_completed_testing_leaves_writing_to_run_single(self):
        with patch.object(app, "write_outputs") as write_outputs:
            self.assertIs(self.perform_q_learning(Mock()), self.learner)
        write_outputs.assert_not_called()

    def test_run_single_writes_outputs_once_after_testing(self):
        graph = self.learner.operation_graph
        with (
            patch.object(app, "construct_db_dir"),
            patch.object(app, "EmbeddingModel"),
            patch.object(app.AutoRestTest, "generate_graph", return_value=graph),
            patch.object(
                app.AutoRestTest, "perform_q_learning", return_value=self.learner
            ),
            patch.object(app.AutoRestTest, "print_performance"),
            patch.object(app, "write_outputs") as write_outputs,
        ):
            app.AutoRestTest("unused", CONFIG, self.tui).run_single("probe", ".json")
        write_outputs.assert_called_once_with(self.learner, "probe", graph.spec_parser)


class WriteJsonTests(unittest.TestCase):
    def test_an_interrupted_write_keeps_the_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            app.write_json(path, {"version": 1})
            with self.assertRaises(TypeError):
                app.write_json(path, {"version": object()})
            self.assertEqual(json.loads(path.read_text()), {"version": 1})
            self.assertEqual(
                [p.name for p in Path(directory).iterdir()], ["report.json"]
            )


if __name__ == "__main__":
    unittest.main()
