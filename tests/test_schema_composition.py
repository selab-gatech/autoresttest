import json
import tempfile
import unittest
from pathlib import Path

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.specification import SpecificationParser

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "probe", "version": "1"},
    "paths": {
        "/visits": {
            "post": {
                "operationId": "addVisit",
                "requestBody": {
                    "description": "The visit",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Visit"}}
                    },
                },
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Visit"},
                                }
                            }
                        },
                    }
                },
            }
        }
    },
    "components": {
        "schemas": {
            "Base": {
                "type": "object",
                "description": "Base fields",
                "properties": {"date": {"type": "string", "format": "date"}},
                "required": ["date"],
            },
            "VisitFields": {
                "allOf": [
                    {"$ref": "#/components/schemas/Base"},
                    {
                        "properties": {"description": {"type": "string"}},
                        "required": ["description"],
                    },
                ]
            },
            "Visit": {
                "description": "A booking for a vet visit.",
                "allOf": [
                    {"$ref": "#/components/schemas/VisitFields"},
                    {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "readOnly": True},
                            "petId": {
                                "oneOf": [
                                    {"type": "integer", "minimum": 0},
                                    {"type": "string"},
                                ]
                            },
                            "vet": {
                                "anyOf": [
                                    {"$ref": "#/components/schemas/Base"},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["petId", "date"],
                    },
                ],
            },
        }
    },
}


class SchemaCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = apply_config_overrides(
            {"spec": {"strict_validation": False}},
            load_config(ROOT / "configurations.toml.example"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(SPEC))
            parser = SpecificationParser(str(path), "probe", config=config)
            cls.operation = parser.parse_specification()["addVisit"]

    def test_all_of_is_merged_recursively(self):
        body = self.operation.request_body["application/json"]
        self.assertEqual(body.type, "object")
        self.assertEqual(
            set(body.properties), {"date", "description", "id", "petId", "vet"}
        )
        self.assertEqual(body.required, ["date", "description", "petId"])
        self.assertEqual(body.properties["date"].format, "date")
        self.assertTrue(body.properties["id"].read_only)

    def test_schema_keys_take_precedence_over_sub_schemas(self):
        body = self.operation.request_body["application/json"]
        # The request body's own description is passed in; the response keeps Visit's.
        self.assertEqual(body.description, "The visit")
        items = self.operation.responses["200"].content["application/json"].items
        self.assertEqual(items.description, "A booking for a vet visit.")
        self.assertEqual(set(items.properties), set(body.properties))

    def test_one_of_and_any_of_use_the_first_option(self):
        properties = self.operation.request_body["application/json"].properties
        self.assertEqual(properties["petId"].type, "integer")
        self.assertEqual(properties["petId"].minimum, 0)
        self.assertEqual(properties["vet"].type, "object")
        self.assertEqual(set(properties["vet"].properties), {"date"})


if __name__ == "__main__":
    unittest.main()
