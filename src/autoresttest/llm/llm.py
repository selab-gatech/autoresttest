import os
import threading
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from autoresttest.config import get_config
from autoresttest.prompts.system_prompts import DEFAULT_SYSTEM_MESSAGE
from autoresttest.utils import encode_dictionary

CONFIG = get_config()

load_dotenv()


@dataclass
class TokenCounter:
    input_tokens: int = 0
    output_tokens: int = 0


class LanguageModel:
    input_tokens = 0
    output_tokens = 0
    cache = {}

    # Thread-safety locks for parallel value generation
    _cache_lock = threading.RLock()
    _token_lock = threading.RLock()

    @staticmethod
    def get_tokens() -> TokenCounter:
        return TokenCounter(
            input_tokens=LanguageModel.input_tokens,
            output_tokens=LanguageModel.output_tokens,
        )

    def __init__(
        self,
        engine=CONFIG.openai_llm_engine,
        temperature=CONFIG.creative_temperature,
        max_tokens=CONFIG.llm_max_tokens,
    ):
        self.api_key = os.getenv("API_KEY")
        if self.api_key is None or self.api_key.strip() == "":
            raise ValueError(
                "API key is required for OpenAI language model, found None or empty string."
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=CONFIG.llm_api_base,
            timeout=CONFIG.llm.timeout_seconds,
            max_retries=2,
        )
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _generate_cache_key(self, user_message, system_message, json_mode):
        key_data = {
            "user_message": user_message,
            "system_message": system_message,
            "json_mode": json_mode,
            "engine": self.engine,
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
        except Exception:
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
            return ""
        content = response.choices[0].message.content
        result = content.strip() if content else ""

        # Thread-safe cache write
        with LanguageModel._cache_lock:
            LanguageModel.cache[cache_key] = result

        return result
