import json
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.graph import OperationGraph, RequestGenerator
from autoresttest.llm import LanguageModel
from autoresttest.llm.llm import TokenCounter
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
        self.addCleanup(LanguageModel.cache.clear)

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

    def test_responses_without_choices_are_counted(self):
        model = self.failing_model(None)
        model.client.chat.completions.create.side_effect = None
        model.client.chat.completions.create.return_value = SimpleNamespace(
            usage=None, choices=[]
        )
        with patch("builtins.print"):
            self.assertEqual(model.query("prompt"), "")
        self.assertEqual(LanguageModel.get_failures(), {"NoChoices": 1})

    def replying_model(self, *messages):
        model = self.failing_model(None)
        model.client.chat.completions.create.side_effect = [
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=message, finish_reason="length")],
            )
            for message in messages
        ]
        return model

    def test_empty_content_is_a_failure_and_is_not_cached(self):
        model = self.replying_model(
            SimpleNamespace(content=None),
            SimpleNamespace(content="  \n"),
            None,
            SimpleNamespace(content='{"id": 1}'),
        )
        with (
            patch.object(LanguageModel, "successful_queries", 0),
            patch("builtins.print") as printed,
        ):
            # The same prompt is sent again each time, as nothing was cached.
            results = [model.query("prompt") for _ in range(4)]
            self.assertEqual(LanguageModel.successful_queries, 1)
        self.assertEqual(results, ["", "", "", '{"id": 1}'])
        self.assertEqual(LanguageModel.get_failures(), {"EmptyContent": 3})
        self.assertIn("finish_reason=length", printed.call_args_list[0].args[0])
        self.assertEqual(model.query("prompt"), '{"id": 1}')  # now from the cache

    def test_a_reply_without_a_message_is_reported(self):
        model = self.replying_model(None)
        with patch("builtins.print") as printed:
            self.assertEqual(model.query("prompt"), "")
        self.assertEqual(LanguageModel.get_failures(), {"EmptyContent": 1})
        self.assertIn(
            "the response has no message, finish_reason=length",
            printed.call_args.args[0],
        )

    def test_only_empty_replies_count_as_no_output(self):
        model = self.replying_model(SimpleNamespace(content=""))
        with (
            patch.object(LanguageModel, "successful_queries", 0),
            patch("builtins.print"),
        ):
            model.query("prompt")
            self.assertTrue(LanguageModel.produced_no_output())

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


def completion(**changes):
    reply = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"id": 1}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    reply.update(changes)
    return json.dumps(reply)


class MalformedReplyTests(unittest.TestCase):
    """Replies sent through the real SDK, which returns unexpected bodies unvalidated."""

    def setUp(self):
        for patcher in (
            patch.object(LanguageModel, "failures", Counter()),
            patch.object(LanguageModel, "cache", {}),
            patch.object(LanguageModel, "successful_queries", 0),
            patch.object(LanguageModel, "input_tokens", 0),
            patch.object(LanguageModel, "output_tokens", 0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        print_patcher = patch("builtins.print")
        self.printed = print_patcher.start()
        self.addCleanup(print_patcher.stop)

    def query(self, content_type, body):
        """Send one query whose 200 reply has this body; return the result and call count."""
        calls = []

        def reply(request):
            calls.append(request)
            return httpx.Response(
                200, headers={"content-type": content_type}, content=body.encode()
            )

        http_client = httpx.Client(transport=httpx.MockTransport(reply))
        self.addCleanup(http_client.close)
        model = LanguageModel(config=CONFIG)
        model.client = OpenAI(
            api_key="unused",
            base_url=CONFIG.llm_api_base,
            max_retries=0,
            http_client=http_client,
        )
        result = model.query("prompt", json_mode=True)
        model.query("prompt", json_mode=True)  # a failure is not cached
        return result, len(calls)

    def test_a_non_json_reply_is_a_failure(self):
        # Before the fix the SDK's str reached response.usage and raised AttributeError.
        result, calls = self.query("text/html", "<html>Sign in to the proxy</html>")
        self.assertEqual((result, calls), ("", 2))
        self.assertEqual(LanguageModel.get_failures(), {"InvalidResponse": 2})
        self.assertIn("<html>Sign in", self.printed.call_args_list[0].args[0])
        self.assertTrue(LanguageModel.produced_no_output())

    def test_malformed_completions_are_failures(self):
        cases = [
            (
                "application/json",
                json.dumps(["not", "a", "completion"]),
                "InvalidResponse",
            ),
            ("application/json", "null", "InvalidResponse"),
            ("application/json", completion(choices=[None]), "NoChoices"),
            ("application/json", completion(choices={"index": 0}), "NoChoices"),
            (
                "application/json",
                completion(
                    choices=[{"index": 0, "message": {"content": [{"text": "x"}]}}]
                ),
                "InvalidResponse",
            ),
        ]
        for content_type, body, reason in cases:
            with self.subTest(body=body[:60]):
                LanguageModel.failures.clear()
                self.assertEqual(self.query(content_type, body), ("", 2))
                self.assertEqual(LanguageModel.get_failures(), {reason: 2})

    def test_an_error_object_keeps_its_message(self):
        body = json.dumps({"error": {"message": "Upstream quota exceeded"}})
        self.assertEqual(self.query("application/json", body), ("", 2))
        self.assertEqual(LanguageModel.get_failures(), {"NoChoices": 2})
        self.assertIn("Upstream quota exceeded", self.printed.call_args_list[0].args[0])

    def test_malformed_token_counts_are_ignored(self):
        body = completion(usage={"prompt_tokens": "3", "completion_tokens": 4.5})
        self.assertEqual(self.query("application/json", body), ('{"id": 1}', 1))
        self.assertEqual(LanguageModel.get_tokens(), TokenCounter(0, 0))

    def test_a_valid_reply_is_counted_and_cached(self):
        self.assertEqual(self.query("application/json", completion()), ('{"id": 1}', 1))
        self.assertEqual(LanguageModel.successful_queries, 1)
        self.assertEqual(LanguageModel.get_tokens(), TokenCounter(3, 4))


if __name__ == "__main__":
    unittest.main()
