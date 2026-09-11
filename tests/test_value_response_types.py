import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from autoresttest.config import load_config
from autoresttest.graph import RequestGenerator
from autoresttest.llm import SmartValueGenerator
from autoresttest.models import OperationProperties, ParameterProperties, SchemaProperties
from autoresttest.utils import attempt_fix_json


class ValueResponseTypeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            Path(__file__).resolve().parents[1] / "configurations.toml.example"
        )
        self.operation = OperationProperties(
            operation_id="probe",
            endpoint_path="/items",
            http_method="post",
            parameters={
                (name, "query"): ParameterProperties(
                    name=name, in_value="query", schema=SchemaProperties(type="string")
                )
                for name in ("id", "tag")
            },
            request_body={"application/json": SchemaProperties(type="object")},
        )
        with patch("autoresttest.llm.value_generator.LanguageModel"):
            self.generator = SmartValueGenerator(self.operation, config=self.config)

    def reply(self, value):
        self.generator.language_model.query.return_value = json.dumps(value)

    def test_wrong_parameter_containers_are_ignored(self):
        for value in (None, [], ["id"], "id", 1, True):
            with self.subTest(value=value):
                self.reply({"parameters": value})
                self.assertEqual(self.generator.generate_parameters(), {})
                self.assertEqual(self.generator.generate_value_agent_params(2), {})
                self.assertEqual(
                    self.generator.generate_informed_value_agent_params(2, []), {}
                )

    def test_wrong_value_containers_do_not_discard_valid_siblings(self):
        for value in (None, [], ["bad"], "bad", 1, True):
            with self.subTest(value=value):
                self.reply({"parameters": {"id::query": value, "tag": {"1": "ok"}}})
                self.assertEqual(
                    self.generator.generate_value_agent_params(2),
                    {("tag", "query"): ["ok"]},
                )

    def test_wrong_body_value_containers_are_ignored(self):
        for value in (None, [], ["body"], "body", 1, True):
            with self.subTest(value=value):
                self.reply({"request_body": value})
                self.assertEqual(
                    self.generator.generate_value_agent_body(2), {"application/json": []}
                )
                self.assertEqual(
                    self.generator.generate_informed_value_agent_body(2, []),
                    {"application/json": []},
                )

    def test_valid_parameter_and_body_values_preserve_json_types(self):
        values = [None, False, 0, "", [], {}, {"nested": [1]}]
        indexed_values = {str(i): value for i, value in enumerate(values)}
        self.reply({"parameters": {"id::query": indexed_values}})
        self.assertEqual(
            self.generator.generate_value_agent_params(len(values)),
            {("id", "query"): values},
        )
        self.reply({"request_body": indexed_values})
        self.assertEqual(
            self.generator.generate_value_agent_body(len(values)),
            {"application/json": values},
        )

    def test_wrong_authentication_container_does_not_stop_header_initialization(self):
        request_generator = RequestGenerator(
            SimpleNamespace(config=self.config), "http://example.invalid"
        )
        for value in (None, [], ["username"], "username", 1, True):
            with self.subTest(value=value):
                self.reply({"authentication_parameters": value})
                with patch(
                    "autoresttest.graph.request_generator.SmartValueGenerator",
                    return_value=self.generator,
                ):
                    self.assertEqual(
                        request_generator.get_auth_info(
                            SimpleNamespace(operation_properties=self.operation)
                        ),
                        [],
                    )

    def test_valid_authentication_fields_are_preserved(self):
        fields = {"query_parameters": {"username": "user", "password": "pass"}}
        self.reply({"authentication_parameters": fields})
        self.assertEqual(self.generator.determine_auth_params(), fields)

    def test_excessive_json_values_use_existing_repair_fallback(self):
        for content in ("[" * 2000 + "0" + "]" * 2000, "9" * 5000):
            with self.subTest(content=content[:20]), patch(
                "autoresttest.llm.value_generator.attempt_fix_json", return_value={}
            ) as repair:
                self.generator.language_model.query.return_value = content
                self.assertEqual(self.generator.generate_parameters(), {})
                repair.assert_called_once_with(content, config=self.config)

    def test_failed_json_repair_does_not_raise_for_excessive_values(self):
        for content in ("[" * 2000 + "0" + "]" * 2000, "9" * 5000):
            with self.subTest(content=content[:20]), patch(
                "autoresttest.llm.LanguageModel",
                return_value=Mock(query=Mock(return_value=content)),
            ), patch("builtins.print"):
                self.assertEqual(attempt_fix_json("{", config=self.config), {})
