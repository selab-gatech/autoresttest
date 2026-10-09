import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.marl.marl import QLearning
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
)
from autoresttest.utils import split_parameter_values

OPERATION = OperationProperties(
    operation_id="probe",
    endpoint_path="/items/{id}",
    http_method="get",
    parameters={
        ("id", "path"): ParameterProperties(
            name="id", in_value="path", schema=SchemaProperties(type="integer")
        ),
        ("limit", "query"): ParameterProperties(
            name="limit", in_value="query", schema=SchemaProperties(type="integer")
        ),
        ("X-Version", "header"): ParameterProperties(
            name="X-Version", in_value="header", schema=SchemaProperties(type="integer")
        ),
    },
)


def make_learner():
    config = apply_config_overrides(
        {"agents": {"header": {"enabled": False}}},
        load_config(Path(__file__).resolve().parents[1] / "configurations.toml.example"),
    )
    graph = SimpleNamespace(
        config=config,
        request_generator=SimpleNamespace(api_url="http://example.invalid"),
        operation_nodes={
            "probe": SimpleNamespace(operation_properties=OPERATION, outgoing_edges=[])
        },
        operation_edges=[],
    )
    return QLearning(graph, time_duration=1, mutation_rate=0)


def send(learner, parameters):
    response = requests.Response()
    response.status_code, response._content = 200, b"{}"
    with patch("requests.get", return_value=response) as get:
        learner.send_operation(OPERATION, parameters, None, None)
    return get.call_args


class SplitParameterValuesTests(unittest.TestCase):
    def test_undefined_keys_are_dropped_by_default(self):
        path, query, header, cookie = split_parameter_values(
            OPERATION.parameters, {("limit", "header"): 5, ("abc", "query"): 1}
        )
        self.assertEqual((path, query, header, cookie), ({}, {}, {}, {}))

    def test_undefined_keys_are_sent_where_their_key_says(self):
        path, query, header, cookie = split_parameter_values(
            OPERATION.parameters,
            {
                ("id", "path"): 7,
                ("limit", "header"): 5,
                ("abc", "query"): [1, 2],
                ("zz", "cookie"): {"a": 1},
                ("p", "path"): 3,
            },
            include_undefined=True,
        )
        self.assertEqual(path, {"id": 7})
        self.assertEqual(query, {"abc": [1, 2]})
        self.assertEqual(header, {"limit": "5"})
        self.assertEqual(cookie, {"zz": '{"a": 1}'})

    def test_header_and_cookie_values_are_strings(self):
        _, _, header, _ = split_parameter_values(
            OPERATION.parameters, {("X-Version", "header"): True}
        )
        self.assertEqual(header, {"X-Version": "true"})


class MutatedRequestTests(unittest.TestCase):
    def test_send_operation_sends_mutated_locations_and_names(self):
        call = send(
            make_learner(),
            {
                ("id", "path"): 3,
                ("limit", "cookie"): 10,
                ("X-Version", "query"): 2,
                ("Rnd", "header"): 1.5,
            },
        )
        self.assertEqual(call.args[0], "http://example.invalid/items/3")
        self.assertEqual(call.kwargs["params"], {"X-Version": 2})
        self.assertEqual(call.kwargs["cookies"], {"limit": "10"})
        self.assertEqual(call.kwargs["headers"]["Rnd"], "1.5")

    def test_mutated_requests_reach_the_api(self):
        learner = make_learner()
        random.seed(0)
        moved = renamed = 0
        for _ in range(300):
            parameters = {("limit", "query"): 10, ("X-Version", "header"): 1}
            parameters, _, _, _, mutated_names = learner.mutate_values(
                OPERATION, parameters, None, None
            )
            call = send(learner, parameters)
            sent = (
                set(call.kwargs["params"] or {})
                | set(call.kwargs["headers"] or {})
                | set(call.kwargs["cookies"] or {})
            )
            # Every non-null parameter left after mutation is sent somewhere.
            expected = {name for (name, _), value in parameters.items() if value is not None}
            self.assertLessEqual(expected, sent)
            if mutated_names:
                renamed += 1
            elif "limit" in (call.kwargs["headers"] or {}) or "limit" in (
                call.kwargs["cookies"] or {}
            ):
                moved += 1
        self.assertGreater(moved, 0)
        self.assertGreater(renamed, 0)


if __name__ == "__main__":
    unittest.main()
