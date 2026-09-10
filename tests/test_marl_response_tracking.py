import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from autoresttest.agents.parameter_agent import ParameterAction
from autoresttest.marl.marl import QLearning
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
)


class RandomDependencyResponseTests(unittest.TestCase):
    def run_requests(self, status_code):
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
        )
        graph = SimpleNamespace(
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
        response._content = b""
        with (
            patch(
                "autoresttest.marl.marl.CONFIG",
                SimpleNamespace(enable_header_agent=False),
            ),
            patch("autoresttest.marl.marl.time.time", side_effect=[0, 0, 0, 2]),
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
