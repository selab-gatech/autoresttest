import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.graph import OperationGraph, RequestGenerator
from autoresttest.llm import LanguageModel
from autoresttest.models import OperationProperties

CONFIG = apply_config_overrides(
    {"llm": {"api_base": "http://127.0.0.1:9/v1", "engine": "local-model"}},
    load_config(Path(__file__).resolve().parents[1] / "configurations.toml.example"),
)


class LlmFailureLoggingTests(unittest.TestCase):
    def setUp(self):
        LanguageModel.failures.clear()
        LanguageModel.cache.clear()
        self.addCleanup(LanguageModel.failures.clear)

    def failing_model(self, error):
        model = LanguageModel(config=CONFIG)
        model.client = Mock()
        model.client.chat.completions.create.side_effect = error
        return model

    def test_failures_are_counted_and_printed_sparsely(self):
        model = self.failing_model(ConnectionError("connection refused"))
        with patch("builtins.print") as printed:
            results = [model.query(f"prompt {i}") for i in range(12)]
        self.assertEqual(results, [""] * 12)
        self.assertEqual(LanguageModel.get_failures(), {"ConnectionError": 12})
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertEqual(len(lines), 2)  # occurrences 1 and 10
        self.assertIn("ConnectionError (occurrence 1)", lines[0])
        self.assertIn("local-model at http://127.0.0.1:9/v1", lines[0])
        self.assertIn("connection refused", lines[0])
        self.assertIn("(occurrence 10)", lines[1])

    def test_empty_responses_are_counted(self):
        model = self.failing_model(None)
        model.client.chat.completions.create.side_effect = None
        model.client.chat.completions.create.return_value = SimpleNamespace(
            usage=None, choices=[]
        )
        with patch("builtins.print"):
            self.assertEqual(model.query("prompt"), "")
        self.assertEqual(LanguageModel.get_failures(), {"NoChoices": 1})

    def test_failed_value_table_operations_are_printed(self):
        graph = OperationGraph("unused", "probe", SimpleNamespace(config=CONFIG), Mock())
        graph.add_operation_node(
            OperationProperties(
                operation_id="listItems", endpoint_path="/items", http_method="get"
            )
        )
        generator = RequestGenerator(graph, "http://example.invalid", is_naive=False)
        mappings = {}
        with (
            patch.object(generator, "create_and_send_request", return_value=None),
            patch(
                "autoresttest.graph.request_generator.SmartValueGenerator",
                side_effect=ValueError("API key is required"),
            ),
            patch("builtins.print") as printed,
        ):
            generator.generate_value_tables_parallel(graph.operation_nodes, mappings)
        self.assertEqual(mappings, {"listItems": {"params": {}, "body": {}}})
        self.assertIn(
            "Value table generation failed for operation listItems: "
            "ValueError: API key is required",
            [call.args[0] for call in printed.call_args_list],
        )


if __name__ == "__main__":
    unittest.main()
