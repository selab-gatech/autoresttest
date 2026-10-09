import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from autoresttest.config import load_config
from autoresttest.llm import SmartValueGenerator
from autoresttest.models import OperationProperties, RequestRequirements

CONFIG = load_config(Path(__file__).resolve().parents[1] / "configurations.toml.example")
OPERATION = OperationProperties(
    operation_id="createPet", endpoint_path="/pets", http_method="post"
)


def generator(requirements=None):
    with patch("autoresttest.llm.value_generator.LanguageModel"):
        return SmartValueGenerator(OPERATION, requirements=requirements, config=CONFIG)


class RequestBodyFieldTests(unittest.TestCase):
    def test_required_body_properties_are_not_requested_again(self):
        requirements = RequestRequirements(
            edge=Mock(), request_body_requirements={"name": "Rex"}
        )
        fields = generator(requirements)._isolate_nonreq_request_body(
            {"type": "object", "properties": {"name": {}, "age": {}}}
        )
        self.assertEqual(list(fields), ["age"])

    def test_array_bodies_list_their_item_properties(self):
        fields = generator()._isolate_nonreq_request_body(
            {"type": "array", "items": {"properties": {"name": {}}}}
        )
        self.assertEqual(list(fields), ["name"])

    def test_bodies_without_properties_list_no_fields(self):
        fields = generator()._isolate_nonreq_request_body(
            {"type": "string", "description": "A note"}
        )
        self.assertEqual(fields, {})


if __name__ == "__main__":
    unittest.main()
