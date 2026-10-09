import sys
import unittest
from pathlib import Path

from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
    ValueAction,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner, run_once  # noqa: E402

JSON = "application/json"
PARAMS = [(name, "query") for name in ("q", "limit", "offset")]
PROPS = ("name", "age", "tag")
OPERATION = OperationProperties(
    operation_id="search",
    endpoint_path="/search",
    http_method="post",
    parameters={
        key: ParameterProperties(
            name=key[0], in_value="query", schema=SchemaProperties(type="string")
        )
        for key in PARAMS
    },
    request_body={
        JSON: SchemaProperties(
            type="object",
            properties={prop: SchemaProperties(type="string") for prop in PROPS},
        )
    },
)


def from_source():
    return {
        "dependent_operation": "listUsers",
        "dependent_val": "name",
        "in_value": "response",
    }


class RandomDependencyRecordingTests(unittest.TestCase):
    def run_random(self, mime_type, body_properties=None):
        learner = make_learner(OPERATION)
        learner.successful_responses["listUsers"] = {"name": ["ada"]}
        run_once(
            learner,
            "search",
            req_params=(("q", "query"),),
            mime_type=mime_type,
            data_source="DEPENDENCY",
            dependency_action=(
                "RANDOM",
                {key: from_source() for key in PARAMS},
                {prop: from_source() for prop in PROPS},
            ),
            value_action=ValueAction(param_mappings=None, body_mappings=None),
            body_properties=body_properties,
        )
        return learner.dependency_agent

    def test_only_sent_parameters_and_properties_are_recorded(self):
        agent = self.run_random(JSON, body_properties=("name",))
        self.assertEqual(list(agent.q_table["search"]["params"]), [("q", "query")])
        self.assertEqual(list(agent.q_table["search"]["body"]), ["name"])
        self.assertEqual(agent.dependencies_discovered, 2)

    def test_no_body_dependencies_are_recorded_without_a_body(self):
        agent = self.run_random(None)
        self.assertEqual(list(agent.q_table["search"]["params"]), [("q", "query")])
        self.assertEqual(agent.q_table["search"]["body"], {})

    def test_repeated_dependencies_are_counted_once(self):
        agent = make_learner(OPERATION).dependency_agent
        for _ in range(3):
            agent.add_new_dependency(
                "search", "params", ("q", "query"), "listUsers", "response", "name"
            )
        self.assertEqual(agent.dependencies_discovered, 1)


if __name__ == "__main__":
    unittest.main()
