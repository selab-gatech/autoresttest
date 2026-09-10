import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresttest.config import load_config
from autoresttest.marl import QLearning
from autoresttest.models import OperationProperties, SchemaProperties


class BodySchemaGuardTests(unittest.TestCase):
    def setUp(self):
        self.operation = OperationProperties(
            operation_id="probe", endpoint_path="/items", http_method="post"
        )
        graph = SimpleNamespace(
            config=load_config(
                Path(__file__).resolve().parents[1] / "configurations.toml.example"
            ),
            request_generator=SimpleNamespace(api_url="http://example.invalid"),
            operation_nodes={
                "probe": SimpleNamespace(operation_properties=self.operation)
            },
            operation_edges=[],
        )
        self.learner = QLearning(graph)

    def build_body(self, schema, values):
        self.operation.request_body = {"application/json": schema}
        return self.learner._construct_body(values, "probe", "application/json")

    def test_object_without_properties_does_not_crash_reconstruction(self):
        self.assertEqual(
            self.build_body(SchemaProperties(type="object"), {"id": 7}), {}
        )

    def test_array_without_item_schema_does_not_crash_reconstruction(self):
        self.assertEqual(
            self.build_body(SchemaProperties(type="array"), {"id": 7}), [None]
        )

    def test_nested_object_without_properties_does_not_crash_reconstruction(self):
        self.assertEqual(
            self.build_body(
                SchemaProperties(type="array", items=SchemaProperties(type="object")),
                {"id": 7},
            ),
            [{}],
        )

    def test_declared_properties_are_preserved(self):
        schema = SchemaProperties(
            type="object", properties={"id": SchemaProperties(type="integer")}
        )
        self.assertEqual(self.build_body(schema, {"id": 7, "extra": 8}), {"id": 7})
        self.assertEqual(
            self.build_body(SchemaProperties(type="array", items=schema), {"id": 7}),
            [{"id": 7}],
        )

    def test_unrecognized_schema_type_does_not_crash_mutation(self):
        for schema_type in ("null", "float", "file", "unsupported"):
            with self.subTest(schema_type=schema_type):
                self.learner.get_mutated_value(schema_type)

    def test_mutation_still_excludes_the_original_supported_type(self):
        types = ["integer", "number", "string", "boolean", "array", "object"]
        for schema_type in types:
            with self.subTest(schema_type=schema_type), patch(
                "autoresttest.marl.marl.random.choice", return_value="string"
            ) as choose, patch(
                "autoresttest.marl.marl.identify_generator", return_value=lambda: "mutated"
            ):
                self.learner.get_mutated_value(schema_type)
                choose.assert_called_once_with([t for t in types if t != schema_type])
