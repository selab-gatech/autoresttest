import unittest
from pathlib import Path

from autoresttest.config import load_config
from autoresttest.llm import SmartValueGenerator
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
)
from autoresttest.utils import remove_nulls


class RemoveNullsTests(unittest.TestCase):
    def test_falsy_spec_values_are_kept(self):
        self.assertEqual(
            remove_nulls(
                {
                    "type": "integer",
                    "minimum": 0,
                    "default": False,
                    "example": "",
                    "description": None,
                    "enum": [],
                    "properties": {"a": None},
                    "items": {"maxLength": 0, "pattern": None},
                }
            ),
            {
                "type": "integer",
                "minimum": 0,
                "default": False,
                "example": "",
                "items": {"maxLength": 0},
            },
        )

    def test_lists_keep_falsy_items(self):
        self.assertEqual(remove_nulls([0, False, None, ""]), [0, False, ""])

    def test_prompt_parameters_include_zero_and_false_constraints(self):
        operation = OperationProperties(
            operation_id="probe",
            endpoint_path="/items",
            http_method="get",
            parameters={
                ("page", "query"): ParameterProperties(
                    name="page",
                    in_value="query",
                    required=False,
                    schema=SchemaProperties(type="integer", minimum=0, default=0),
                )
            },
        )
        config = load_config(
            Path(__file__).resolve().parents[1] / "configurations.toml.example"
        )
        generator = SmartValueGenerator(operation, config=config)
        schema = generator.parameters["page::query"]["schema"]
        self.assertEqual(schema["minimum"], 0)
        self.assertEqual(schema["default"], 0)
        self.assertIs(generator.parameters["page::query"]["required"], False)


if __name__ == "__main__":
    unittest.main()
