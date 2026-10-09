import unittest
from pathlib import Path
from unittest.mock import patch

from autoresttest.config import load_config
from autoresttest.llm import SmartValueGenerator
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
)
from autoresttest.prompts import PARAMETERS_GEN_PROMPT

OPERATION = OperationProperties(
    operation_id="getUser",
    endpoint_path="/users/{userId}",
    http_method="get",
    parameters={
        ("userId", "path"): ParameterProperties(
            name="userId", in_value="path", schema=SchemaProperties(type="string")
        ),
        ("limit", "query"): ParameterProperties(
            name="limit", in_value="query", schema=SchemaProperties(type="integer")
        ),
    },
)


class ParameterNameFallbackTests(unittest.TestCase):
    def setUp(self):
        config = load_config(
            Path(__file__).resolve().parents[1] / "configurations.toml.example"
        )
        with patch("autoresttest.llm.value_generator.LanguageModel"):
            self.generator = SmartValueGenerator(OPERATION, config=config)

    def test_plain_and_suffixed_names_are_accepted(self):
        self.assertEqual(
            self.generator._validate_parameters(
                {"userId": "u1", "limit::query": 5, "unknown": 1}
            ),
            {("userId", "path"): "u1", ("limit", "query"): 5},
        )

    def test_prompt_asks_for_suffixed_keys(self):
        self.assertIn('"name::query"', PARAMETERS_GEN_PROMPT)
        self.assertIn("Do NOT use plain parameter names", PARAMETERS_GEN_PROMPT)


if __name__ == "__main__":
    unittest.main()
