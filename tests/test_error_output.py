import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresttest.autoresttest import output_errors


class ErrorOutputTests(unittest.TestCase):
    def test_parameterized_errors_preserve_request_values(self):
        errors = {
            "probe": [
                {
                    "parameters": {
                        ("id", "path"): 42,
                        ("id", "query"): None,
                        ("enabled", "query"): False,
                        ("limit", "query"): 0,
                        ("search", "query"): "",
                    },
                    "body": {"application/json": {"value": None, "items": [None, 0]}},
                    "operation_id": "probe",
                }
            ],
            "without_parameters": [
                {
                    "parameters": None,
                    "body": None,
                    "operation_id": "without_parameters",
                }
            ],
            "legacy": [
                {"parameters": {"id": 42}, "body": {}, "operation_id": "legacy"}
            ],
            "unsupported": [
                {"parameters": {}, "body": b"binary", "operation_id": "unsupported"}
            ],
        }
        original = copy.deepcopy(errors)
        with tempfile.TemporaryDirectory() as directory:
            with patch("autoresttest.autoresttest.DATA_ROOT", Path(directory)):
                output_errors(SimpleNamespace(unique_errors=errors), "probe")
            saved = json.loads(
                (Path(directory) / "probe/server_errors.json").read_text()
            )

        expected = copy.deepcopy(original)
        expected["probe"][0]["parameters"] = {
            "id::path": 42,
            "id::query": None,
            "enabled::query": False,
            "limit::query": 0,
            "search::query": "",
        }
        expected["unsupported"] = []
        self.assertEqual(saved, expected)
        self.assertEqual(errors, original)
