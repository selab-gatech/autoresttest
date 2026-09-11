import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from autoresttest.agents.parameter_agent import ParameterAction
from autoresttest.config import apply_config_overrides, load_config
from autoresttest.marl.marl import QLearning
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    ResponseProperties,
    SchemaProperties,
)


class RandomDependencyResponseTests(unittest.TestCase):
    def run_requests(self, status_code, content=b""):
        parameter = ("id", "query")
        operation = OperationProperties(
            operation_id="probe",
            endpoint_path="/items",
            http_method="get",
            parameters={
                parameter: ParameterProperties(
                    name="id",
                    in_value="query",
                    required=True,
                    schema=SchemaProperties(type="integer"),
                )
            },
            responses={
                "200": ResponseProperties(
                    content={
                        "application/json": SchemaProperties(
                            type="object",
                            properties={"id": SchemaProperties(type="integer")},
                        )
                    }
                )
            },
        )
        graph = SimpleNamespace(
            config=apply_config_overrides(
                {"agents": {"header": {"enabled": False}}},
                load_config(
                    Path(__file__).resolve().parents[1] / "configurations.toml.example"
                ),
            ),
            request_generator=SimpleNamespace(api_url="http://example.invalid"),
            operation_nodes={
                "probe": SimpleNamespace(
                    operation_properties=operation, outgoing_edges=[]
                )
            },
            operation_edges=[],
        )
        learner = QLearning(graph, time_duration=1, mutation_rate=0)
        for agent in (
            learner.operation_agent,
            learner.parameter_agent,
            learner.body_object_agent,
            learner.data_source_agent,
            learner.dependency_agent,
        ):
            agent.initialize_q_table()
        learner.data_source_agent.initialize_dependency_source()
        learner.value_agent.q_table = {"probe": {"params": {}, "body": {}}}
        learner.successful_responses["source"] = {"id": [42]}
        dependency = {
            parameter: {
                "dependent_operation": "source",
                "in_value": "response",
                "dependent_val": "id",
            }
        }
        response = requests.Response()
        response.status_code = status_code
        response._content = content
        with (
            patch("autoresttest.marl.marl.time.monotonic", side_effect=[0, 0, 0, 2]),
            patch.object(learner.operation_agent, "get_action", return_value="probe"),
            patch.object(
                learner.parameter_agent,
                "get_action",
                return_value=ParameterAction((parameter,), None),
            ),
            patch.object(
                learner.data_source_agent, "get_action", return_value="DEPENDENCY"
            ),
            patch.object(
                learner.dependency_agent,
                "get_action",
                return_value=("RANDOM", dependency, {}),
            ),
            patch.object(
                learner.dependency_agent, "add_new_dependency"
            ) as add_dependency,
            patch.object(learner, "send_operation", return_value=response) as send,
        ):
            learner.execute_operations()

        self.assertEqual(send.call_count, 2)
        self.assertEqual(learner.time_duration, 1)
        self.assertTrue(
            all(call.kwargs["deadline"] == 1 for call in send.call_args_list)
        )
        self.assertEqual(learner.responses[status_code], 2)
        self.assertEqual(learner.operation_response_counter["probe"][status_code], 2)
        self.assertEqual(add_dependency.call_count, 2 if status_code == 200 else 0)
        return learner

    def test_server_errors_are_counted_and_deduplicated(self):
        learner = self.run_requests(500)
        self.assertEqual(learner.errors, {"probe": 2})
        self.assertEqual(
            learner.unique_errors["probe"],
            [
                {
                    "parameters": {("id", "query"): 42},
                    "body": {},
                    "operation_id": "probe",
                }
            ],
        )

    def test_client_errors_are_counted_without_learning_dependencies(self):
        learner = self.run_requests(400)
        self.assertEqual(learner.errors, {})
        self.assertEqual(learner.unique_errors, {})

    def test_successful_responses_still_learn_dependencies(self):
        learner = self.run_requests(200)
        self.assertEqual(learner.errors, {})
        self.assertEqual(learner.successful_parameters["probe"][("id", "query")], [42])

    def test_invalid_json_responses_do_not_stop_testing_or_teach_values(self):
        bodies = [
            b'{"value":"' + b"a" * 16 + b'\x9b"}',
            b"<html>OK</html>",
            b'{"value":',
            b"[" * 2000 + b"0" + b"]" * 2000,
            b'{"value":' + b"9" * 5000 + b"}",
        ]
        for content in bodies:
            with self.subTest(content=content[:40]), patch("builtins.print"):
                learner = self.run_requests(200, content)
                self.assertEqual(learner.successful_responses["probe"], {"id": []})
                self.assertEqual(learner.successful_primitives.get("probe", []), [])
                self.assertEqual(
                    learner.successful_parameters["probe"][("id", "query")], [42]
                )

    def test_valid_json_responses_still_teach_values(self):
        for encoding in ("utf-8", "utf-16", "utf-32"):
            with self.subTest(encoding=encoding):
                learner = self.run_requests(200, '{"id": 7}'.encode(encoding))
                self.assertEqual(learner.successful_responses["probe"]["id"], [7])

    def test_valid_json_primitives_still_teach_values(self):
        for content, expected in ((b"null", [None]), (b"[7, false]", [7, False])):
            with self.subTest(content=content):
                learner = self.run_requests(200, content)
                self.assertEqual(learner.successful_primitives["probe"], expected)
