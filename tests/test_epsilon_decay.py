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
    SchemaProperties,
)


def run_with_clock(times, time_duration=100):
    """Run the testing loop with time.monotonic() returning `times` in order.

    The first value is the start time; each further value below the deadline runs
    one request. Returns the parameter agent's epsilon after each decay.
    """
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
        config=apply_config_overrides(
            {"agents": {"header": {"enabled": False}}},
            load_config(
                Path(__file__).resolve().parents[1] / "configurations.toml.example"
            ),
        ),
        request_generator=SimpleNamespace(api_url="http://example.invalid"),
        operation_nodes={
            "probe": SimpleNamespace(operation_properties=operation, outgoing_edges=[])
        },
        operation_edges=[],
    )
    learner = QLearning(
        graph, epsilon=1.0, time_duration=time_duration, mutation_rate=0
    )
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
    response = requests.Response()
    response.status_code, response._content = 400, b""

    epsilons = []
    decay = learner.epsilon_decay

    def record_decay(amount):
        decay(amount)
        epsilons.append(learner.parameter_agent.epsilon)

    with (
        patch("autoresttest.marl.marl.time.monotonic", side_effect=times),
        patch.object(learner, "epsilon_decay", side_effect=record_decay),
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
            learner.dependency_agent, "get_action", return_value=("RANDOM", {}, {})
        ),
        patch.object(learner, "send_operation", return_value=response),
    ):
        learner.execute_operations()
    return epsilons


class EpsilonDecayTests(unittest.TestCase):
    def test_epsilon_reaches_its_floor_at_70_percent_of_the_budget(self):
        epsilons = run_with_clock([0, 0, 7, 35, 70, 100])
        for actual, expected in zip(epsilons, [1.0, 0.91, 0.55, 0.1]):
            self.assertAlmostEqual(actual, expected)

    def test_epsilon_does_not_depend_on_the_request_rate(self):
        # 1000 fast requests in the first second of a 100 s budget.
        times = [0] + [i / 1000 for i in range(1000)] + [100]
        epsilons = run_with_clock(times)
        self.assertEqual(len(epsilons), 1000)
        self.assertAlmostEqual(epsilons[-1], 1.0 - 0.9 / 70 * 0.999)


if __name__ == "__main__":
    unittest.main()
