import tempfile
import unittest
from pathlib import Path

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.specification import SpecificationParser
from autoresttest.utils import get_accept_header

ROOT = Path(__file__).resolve().parents[1]

# Unquoted status codes are integer keys in YAML.
SPEC = """
openapi: 3.0.0
info: {title: probe, version: "1"}
paths:
  /items:
    get:
      operationId: listItems
      responses:
        200:
          description: ok
          content:
            application/json:
              schema: {type: array, items: {type: string}}
        404:
          description: missing
"""


class StatusCodeKeyTests(unittest.TestCase):
    def parse(self, strict_validation):
        config = apply_config_overrides(
            {"spec": {"strict_validation": strict_validation}},
            load_config(ROOT / "configurations.toml.example"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.yaml"
            path.write_text(SPEC)
            parser = SpecificationParser(str(path), "probe", config=config)
            return parser, parser.parse_specification()["listItems"]

    def test_unquoted_status_codes_parse_as_strings(self):
        for strict_validation in (True, False):
            with self.subTest(strict_validation=strict_validation):
                _, operation = self.parse(strict_validation)
                self.assertEqual(list(operation.responses), ["200", "404"])
                self.assertEqual(operation.responses["200"].status_code, "200")
                self.assertEqual(get_accept_header(operation.responses), "application/json")

    def test_integer_keys_are_stringified_when_processing_responses(self):
        parser, _ = self.parse(False)
        responses = parser.process_responses({201: {"description": "created"}})
        self.assertEqual(list(responses), ["201"])
        self.assertEqual(responses["201"].status_code, "201")


if __name__ == "__main__":
    unittest.main()
