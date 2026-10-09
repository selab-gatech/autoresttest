import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from autoresttest import autoresttest as app
from autoresttest.config import apply_config_overrides, load_config
from autoresttest.graph import OperationGraph, RequestGenerator
from autoresttest.marl import QLearning
from autoresttest.models import OperationProperties
from autoresttest.specification import SpecificationParser

ROOT = Path(__file__).resolve().parents[1]


def write_spec(directory, servers):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "probe", "version": "1"},
        "paths": {
            "/items": {
                "get": {
                    "operationId": "listItems",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    if servers is not None:
        spec["servers"] = servers
    path = Path(directory) / "spec.json"
    path.write_text(json.dumps(spec))
    return path


class ApiUrlTests(unittest.TestCase):
    def setUp(self):
        base = load_config(ROOT / "configurations.toml.example")
        self.config = apply_config_overrides(
            {"api": {"override_url": True, "host": "target", "port": 9966}}, base
        )

    def init_graph(self, servers):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = write_spec(directory, servers)
            parser = SpecificationParser(str(spec_path), "probe", config=self.config)
            with (
                patch.object(app, "construct_db_dir"),
                patch.object(app, "SpecificationParser", return_value=parser),
            ):
                runner = app.AutoRestTest("unused", self.config, Mock())
                return runner.init_graph("probe", str(spec_path), Mock())

    def test_override_url_keeps_the_spec_base_path(self):
        graph = self.init_graph([{"url": "https://example.com/petclinic/api/"}])
        self.assertEqual(
            graph.request_generator.api_url, "http://target:9966/petclinic/api"
        )

    def test_override_url_without_a_base_path_has_no_trailing_slash(self):
        for servers in (None, [{"url": "http://localhost:8080/"}]):
            with self.subTest(servers=servers):
                graph = self.init_graph(servers)
                self.assertEqual(graph.request_generator.api_url, "http://target:9966")

    def test_override_url_accepts_relative_server_urls(self):
        graph = self.init_graph([{"url": "/api/v3"}])
        self.assertEqual(graph.request_generator.api_url, "http://target:9966/api/v3")

    def test_templated_server_paths_are_not_sent_literally(self):
        graph = self.init_graph(
            [
                {
                    "url": "http://localhost/{basePath}",
                    "variables": {"basePath": {"default": "v2"}},
                }
            ]
        )
        self.assertEqual(graph.request_generator.api_url, "http://target:9966")

    def test_requests_do_not_contain_a_double_slash(self):
        graph = OperationGraph(
            "unused", "probe", SimpleNamespace(config=self.config), Mock()
        )
        operation = OperationProperties(
            operation_id="listItems", endpoint_path="/items", http_method="get"
        )
        graph.add_operation_node(operation)
        graph.assign_request_generator(
            RequestGenerator(graph, "http://localhost:8080/", is_naive=False)
        )
        response = requests.Response()
        response.status_code, response._content = 200, b"{}"
        with patch("requests.get", return_value=response) as send:
            QLearning(graph).send_operation(operation, {}, None, None)
        self.assertEqual(send.call_args.args[0], "http://localhost:8080/items")


if __name__ == "__main__":
    unittest.main()
