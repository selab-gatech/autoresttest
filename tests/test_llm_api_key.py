import os
import unittest
from unittest.mock import patch

from autoresttest.config import apply_config_overrides
from autoresttest.llm import LanguageModel


def config_for(api_base):
    return apply_config_overrides({"llm": {"api_base": api_base}})


class LlmApiKeyTests(unittest.TestCase):
    def setUp(self):
        environ = patch.dict("os.environ")
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop("API_KEY", None)

    def test_local_endpoints_do_not_require_a_key(self):
        for api_base in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
        ):
            with (
                self.subTest(api_base=api_base),
                patch("autoresttest.llm.llm.OpenAI") as client,
            ):
                LanguageModel(config=config_for(api_base))
                self.assertEqual(client.call_args.kwargs["api_key"], "")

    def test_empty_key_sends_no_authorization_header(self):
        model = LanguageModel(config=config_for("http://127.0.0.1:8000/v1"))
        self.assertNotIn("Authorization", model.client.auth_headers)

    def test_remote_endpoints_still_require_a_key(self):
        for api_key in (None, "", "   "):
            if api_key is not None:
                os.environ["API_KEY"] = api_key
            with self.subTest(api_key=api_key), self.assertRaises(ValueError):
                LanguageModel(config=config_for("https://api.openai.com/v1"))

    def test_configured_key_is_used_for_local_and_remote_endpoints(self):
        os.environ["API_KEY"] = "test-key"
        for api_base in ("http://localhost:8000/v1", "https://openrouter.ai/api/v1"):
            with (
                self.subTest(api_base=api_base),
                patch("autoresttest.llm.llm.OpenAI") as client,
            ):
                LanguageModel(config=config_for(api_base))
                self.assertEqual(client.call_args.kwargs["api_key"], "test-key")


if __name__ == "__main__":
    unittest.main()
