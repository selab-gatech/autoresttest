import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tomli
from pydantic import ValidationError

from autoresttest.config import Config, get_config
from autoresttest.config.config import LLMConfig, ApiConfig, _load_raw_config
from autoresttest.llm import LanguageModel
from autoresttest.utils import dispatch_request


class TimeoutConfigTests(unittest.TestCase):
    def test_example_is_valid_and_timeouts_have_backward_compatible_defaults(self):
        example = Path(__file__).resolve().parents[1] / "configurations.toml.example"
        raw = tomli.loads(example.read_text())
        self.assertEqual(Config.model_validate(raw).api.request_timeout_seconds, 30)
        del raw["llm"]["timeout_seconds"]
        del raw["api"]["request_timeout_seconds"]
        config = Config.model_validate(raw)
        self.assertEqual(config.llm.timeout_seconds, 120)
        self.assertEqual(config.api.request_timeout_seconds, 30)

    def test_timeouts_must_be_positive_and_finite(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ApiConfig(request_timeout_seconds=value)
            with self.subTest(value=value), self.assertRaises(ValidationError):
                LLMConfig(
                    engine="test",
                    creative_temperature=1,
                    strict_temperature=1,
                    timeout_seconds=value,
                )

    def test_missing_config_explains_how_to_create_it(self):
        with patch(
            "autoresttest.config.config.CONFIG_PATH",
            Path("/nonexistent/configurations.toml"),
        ):
            with self.assertRaisesRegex(
                FileNotFoundError, "Copy configurations.toml.example"
            ):
                _load_raw_config()


class ApiTimeoutTests(unittest.TestCase):
    def test_all_body_formats_receive_the_configured_timeout(self):
        bodies = [
            None,
            "raw",
            {"application/json": {"id": 1}},
            {"application/json": None},
            {"text/plain": "text"},
            {"multipart/form-data": {"id": 1}},
            {"application/x-www-form-urlencoded": {"id": 1}},
            {"application/octet-stream": b"bytes"},
        ]
        with patch.object(get_config().api, "request_timeout_seconds", 7.5):
            for body in bodies:
                with self.subTest(body=body):
                    send = Mock(return_value=SimpleNamespace(status_code=200))
                    dispatch_request(send, "http://example.invalid", {}, body)
                    self.assertEqual(send.call_args.kwargs["timeout"], 7.5)

    def test_expired_deadline_sends_nothing(self):
        send = Mock()
        with patch("autoresttest.utils.utils.time.monotonic", return_value=10):
            response = dispatch_request(
                send, "http://example.invalid", {}, None, deadline=10
            )
        self.assertIsNone(response)
        send.assert_not_called()

    def test_rate_limit_wait_consumes_remaining_budget_without_another_attempt(self):
        clock = [0.0]
        response = SimpleNamespace(status_code=429, headers={"Retry-After": "1000"})

        def request(*args, **kwargs):
            clock[0] += 1
            return response

        def sleep(seconds):
            clock[0] += seconds

        send = Mock(side_effect=request)
        with (
            patch(
                "autoresttest.utils.utils.time.monotonic", side_effect=lambda: clock[0]
            ),
            patch("autoresttest.utils.utils.time.sleep", side_effect=sleep) as wait,
        ):
            actual = dispatch_request(
                send, "http://example.invalid", {}, None, timeout_seconds=30, deadline=5
            )
        self.assertIs(actual, response)
        send.assert_called_once()
        self.assertEqual(send.call_args.kwargs["timeout"], 5)
        wait.assert_called_once_with(4)
        self.assertEqual(clock[0], 5)

    def test_setup_does_not_wait_for_excessive_retry_after(self):
        response = SimpleNamespace(status_code=429, headers={"Retry-After": "1000"})
        send = Mock(return_value=response)
        with patch("autoresttest.utils.utils.time.sleep") as wait:
            actual = dispatch_request(
                send, "http://example.invalid", {}, None, timeout_seconds=30
            )
        self.assertIs(actual, response)
        send.assert_called_once()
        wait.assert_not_called()

    def test_normal_rate_limit_retry_still_succeeds(self):
        response = SimpleNamespace(status_code=200)
        send = Mock(
            side_effect=[SimpleNamespace(status_code=429, headers={}), response]
        )
        with (
            patch("autoresttest.utils.utils.time.sleep") as wait,
            patch("autoresttest.utils.utils.random.uniform", return_value=0),
        ):
            actual = dispatch_request(
                send, "http://example.invalid", {}, None, timeout_seconds=30
            )
        self.assertIs(actual, response)
        self.assertEqual(send.call_count, 2)
        wait.assert_called_once_with(1)


class LlmTimeoutTests(unittest.TestCase):
    def test_configured_timeout_and_sdk_retry_ownership(self):
        with (
            patch.dict("os.environ", {"API_KEY": "test-key"}),
            patch.object(get_config().llm, "timeout_seconds", 45),
            patch("autoresttest.llm.llm.OpenAI") as client,
            patch.object(LanguageModel, "cache", {}),
        ):
            model = LanguageModel()
            self.assertEqual(client.call_args.kwargs["timeout"], 45)
            self.assertEqual(client.call_args.kwargs["max_retries"], 2)
            create = client.return_value.chat.completions.create
            create.side_effect = TimeoutError("timed out")
            self.assertEqual(model.query("test"), "")
            create.assert_called_once()
