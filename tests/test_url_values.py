import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from autoresttest.config import load_config
from autoresttest.graph import OperationGraph, RequestGenerator
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    RequestData,
    SchemaProperties,
)
from autoresttest.utils import fill_path, split_parameter_values
from autoresttest.utils.utils import _dispatch_request_inner, query_value

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner  # noqa: E402

CONFIG = load_config(
    Path(__file__).resolve().parents[1] / "configurations.toml.example"
)
PATH_VALUES = {("id", "path"): "a/b", ("tags", "path"): ["x", None, "y"]}

OPERATION = OperationProperties(
    operation_id="getItem",
    endpoint_path="/items/{id}/tags/{tags}",
    http_method="get",
    parameters={
        ("id", "path"): ParameterProperties(
            name="id", in_value="path", schema=SchemaProperties(type="string")
        ),
        ("tags", "path"): ParameterProperties(
            name="tags", in_value="path", schema=SchemaProperties(type="array")
        ),
        ("active", "query"): ParameterProperties(
            name="active", in_value="query", schema=SchemaProperties(type="boolean")
        ),
        ("filter", "query"): ParameterProperties(
            name="filter", in_value="query", schema=SchemaProperties(type="object")
        ),
        ("ids", "query"): ParameterProperties(
            name="ids", in_value="query", schema=SchemaProperties(type="array")
        ),
    },
)


def wire_url(url, params):
    return requests.Request("GET", url, params=params).prepare().url


class QueryValueTests(unittest.TestCase):
    def test_values(self):
        cases = [
            (True, "true"),
            (False, "false"),
            (5, 5),
            (1.5, 1.5),
            ("text", "text"),
            (None, None),
            ({"x": 1, "y": [True]}, '{"x":1,"y":[true]}'),
            ([True, 2, "a"], ["true", 2, "a"]),
            ([{"a": 1}, [1, 2]], ['{"a":1}', "[1,2]"]),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(query_value(value), expected)

    def test_query_string_on_the_wire(self):
        _, query, _, _ = split_parameter_values(
            OPERATION.parameters,
            {
                ("active", "query"): True,
                ("filter", "query"): {"x": 1},
                ("ids", "query"): [1, False],
            },
        )
        self.assertEqual(
            wire_url("http://h/items", query),
            "http://h/items?active=true&filter=%7B%22x%22%3A1%7D&ids=1&ids=false",
        )


class FillPathTests(unittest.TestCase):
    def test_values_are_percent_encoded_in_their_segment(self):
        cases = [
            (7, "/items/7"),
            ("a/b#c?d", "/items/a%2Fb%23c%3Fd"),
            ("x y", "/items/x%20y"),
            ("é", "/items/%C3%A9"),
            (True, "/items/true"),
            (["a", "b"], "/items/a,b"),
            ([True, 3], "/items/true,3"),
            (["x,y", "z"], "/items/x%2Cy,z"),
            ([None, 1], "/items/1"),
            ({"k": 1}, "/items/%7B%22k%22%3A1%7D"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(fill_path("/items/{id}", {"id": value}), expected)

    def test_unfilled_and_repeated_placeholders(self):
        self.assertEqual(
            fill_path("/a/{id}/b/{id}/{other}", {"id": "x/y"}),
            "/a/x%2Fy/b/x%2Fy/{other}",
        )


def form_body(payload):
    select_method = Mock()
    _dispatch_request_inner(
        select_method,
        "http://h/form",
        params={},
        body={"application/x-www-form-urlencoded": payload},
        headers={},
        cookies=None,
    )
    data = select_method.call_args.kwargs["data"]
    return requests.Request("POST", "http://h/form", data=data).prepare().body


class FormBodyTests(unittest.TestCase):
    def test_form_fields_use_the_same_values(self):
        payload = {
            "agree": True,
            "meta": {"k": "v"},
            "tags": ["a", False],
            "skip": None,
        }
        self.assertEqual(
            form_body(payload),
            "agree=true&meta=%7B%22k%22%3A%22v%22%7D&tags=a&tags=false",
        )

    def test_an_empty_object_sends_an_empty_form(self):
        self.assertFalse(form_body({}))

    def test_a_non_object_payload_is_sent_as_data(self):
        self.assertEqual(form_body("text"), "data=text")
        self.assertEqual(form_body([{"k": True}]), "k=true")


class SendOperationTests(unittest.TestCase):
    def setUp(self):
        self.response = requests.Response()
        self.response.status_code, self.response._content = 200, b"{}"

    def assert_wire_values(self, call):
        self.assertEqual(call.args[0], "http://example.invalid/items/a%2Fb/tags/x,y")
        self.assertEqual(call.kwargs["params"], {"active": "false"})

    def test_testing_requests_carry_wire_values(self):
        learner = make_learner(OPERATION)
        with patch("requests.get", return_value=self.response) as get:
            learner.send_operation(
                OPERATION, {**PATH_VALUES, ("active", "query"): False}, None, None
            )
        self.assert_wire_values(get.call_args)

    def test_setup_requests_carry_wire_values(self):
        graph = OperationGraph(
            "unused",
            "probe",
            SimpleNamespace(config=CONFIG),
            Mock(),
        )
        graph.add_operation_node(OPERATION)
        generator = RequestGenerator(graph, "http://example.invalid", is_naive=False)
        with patch("requests.get", return_value=self.response) as get:
            generator.send_operation_request(
                RequestData(
                    endpoint_path=OPERATION.endpoint_path,
                    http_method="get",
                    parameters={**PATH_VALUES, ("active", "query"): False},
                    request_body=None,
                    operation_properties=OPERATION,
                )
            )
        self.assert_wire_values(get.call_args)


if __name__ == "__main__":
    unittest.main()
