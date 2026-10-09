import itertools
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
    RequestResponse,
    SchemaProperties,
)


def request_response(status_code):
    response = requests.Response()
    response.status_code, response._content = status_code, b"{}"
    return RequestResponse(
        request=RequestData(
            endpoint_path="/login",
            http_method="get",
            parameters={},
            request_body=None,
            operation_properties=None,
        ),
        response=response,
        response_text="{}",
    )


class DetermineTokensTests(unittest.TestCase):
    def determine_tokens(self, status_code):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configurations.toml.example"
        )
        operation = OperationProperties(
            operation_id="login",
            endpoint_path="/login",
            http_method="get",
            parameters={
                (name, "query"): ParameterProperties(
                    name=name, in_value="query", schema=SchemaProperties(type="string")
                )
                for name in ("username", "password")
            },
        )
        graph = OperationGraph("unused", "probe", SimpleNamespace(config=config), Mock())
        graph.add_operation_node(operation)
        generator = RequestGenerator(graph, "http://example.invalid", is_naive=False)
        with (
            patch.object(
                generator,
                "create_and_send_request",
                return_value=request_response(status_code),
            ),
            patch.object(
                generator,
                "send_operation_request",
                return_value=request_response(status_code),
            ),
            patch("autoresttest.graph.request_generator.SmartValueGenerator") as llm,
            # Skip the 30 s token search loop after value generation.
            patch(
                "autoresttest.graph.request_generator.time.time",
                side_effect=itertools.count(0, 100),
            ),
        ):
            for method in (
                "generate_informed_value_agent_params",
                "generate_value_agent_params",
                "generate_informed_value_agent_body",
                "generate_value_agent_body",
            ):
                getattr(llm.return_value, method).return_value = {}
            generator.determine_tokens(
                graph.operation_nodes["login"],
                {"username": "username", "password": "password"},
                [],
            )
        return llm.return_value

    def test_failed_responses_inform_value_generation(self):
        llm = self.determine_tokens(401)
        llm.generate_value_agent_params.assert_not_called()
        responses = llm.generate_informed_value_agent_params.call_args.kwargs[
            "responses"
        ]
        self.assertEqual([r.response.status_code for r in responses], [401] * 3)

    def test_successful_responses_use_uninformed_generation(self):
        llm = self.determine_tokens(200)
        llm.generate_informed_value_agent_params.assert_not_called()
        llm.generate_value_agent_params.assert_called_once()


if __name__ == "__main__":
    unittest.main()
