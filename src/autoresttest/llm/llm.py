import math
import os
import threading
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from autoresttest.config import Config, get_config
from autoresttest.prompts.system_prompts import DEFAULT_SYSTEM_MESSAGE
from autoresttest.utils import encode_dictionary


load_dotenv()

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_endpoint(api_base: str) -> bool:
    return urlparse(api_base).hostname in LOCAL_HOSTS


@dataclass
class TokenCounter:
    input_tokens: int = 0
    output_tokens: int = 0


class LanguageModel:
    input_tokens = 0
    output_tokens = 0
    cache = {}

    # Failed queries by reason, e.g. {"APIConnectionError": 240}
    failures: Counter = Counter()

    # Thread-safety locks for parallel value generation
    _cache_lock = threading.RLock()
    _token_lock = threading.RLock()
    _failure_lock = threading.Lock()

    @staticmethod
    def get_tokens() -> TokenCounter:
        return TokenCounter(
            input_tokens=LanguageModel.input_tokens,
            output_tokens=LanguageModel.output_tokens,
        )

    @staticmethod
    def get_failures() -> dict[str, int]:
        with LanguageModel._failure_lock:
            return dict(LanguageModel.failures)

    def _report_failure(self, reason: str, detail: str) -> None:
        """Print the first failure of each reason, then its 10th, 100th, ... occurrence."""
        with LanguageModel._failure_lock:
            LanguageModel.failures[reason] += 1
            count = LanguageModel.failures[reason]
        if count == 10 ** int(math.log10(count)):
            print(
                f"[LLM] Query failed: {reason} (occurrence {count}) from model "
                f"{self.engine} at {self.config.llm_api_base}: {detail[:300]}"
            )

    def __init__(
        self,
        engine=None,
        temperature=None,
        max_tokens=None,
        config: Config | None = None,
    ):
        self.config = config if config is not None else get_config()
        self.api_key = os.getenv("API_KEY") or ""
        if self.api_key.strip() == "":
            if not is_local_endpoint(self.config.llm_api_base):
                raise ValueError(
                    "API key is required for OpenAI language model, found None or empty string."
                )
            # Local servers usually need no key; with an empty key the SDK omits the auth header.
            self.api_key = ""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.config.llm_api_base,
            timeout=self.config.llm.timeout_seconds,
            max_retries=2,
        )
        self.engine = engine if engine is not None else self.config.openai_llm_engine
        self.temperature = (
            temperature if temperature is not None else self.config.creative_temperature
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None else self.config.llm_max_tokens
        )

    def _generate_cache_key(self, user_message, system_message, json_mode):
        key_data = {
            "user_message": user_message,
            "system_message": system_message,
            "json_mode": json_mode,
            "engine": self.engine,
            "api_base": self.config.llm_api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return encode_dictionary(key_data)

    def query(
        self, user_message, system_message=DEFAULT_SYSTEM_MESSAGE, json_mode=False
    ) -> str:
        cache_key = self._generate_cache_key(user_message, system_message, json_mode)

        # Thread-safe cache read
        with LanguageModel._cache_lock:
            if cache_key in LanguageModel.cache:
                return LanguageModel.cache[cache_key]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        kwargs = {
            "model": self.engine,
            "messages": messages,
            "temperature": self.temperature,
        }

        if self.max_tokens != -1:
            kwargs["max_tokens"] = self.max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # The SDK owns retries; wrapping it in another retry loop multiplies attempts.
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self._report_failure(type(exc).__name__, str(exc))
            return ""

        input_tokens = 0
        output_tokens = 0
        if response.usage is not None:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        # print(f"[LLM] Input tokens: {input_tokens}, Output tokens: {output_tokens}")

        # Thread-safe token updates
        with LanguageModel._token_lock:
            LanguageModel.input_tokens += input_tokens
            LanguageModel.output_tokens += output_tokens

        if not response.choices:
            self._report_failure("NoChoices", "the response has no choices")
            return ""
        content = response.choices[0].message.content
        result = content.strip() if content else ""

        # Thread-safe cache write
        with LanguageModel._cache_lock:
            LanguageModel.cache[cache_key] = result

        return result
