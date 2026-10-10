import io
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.tui.config_wizard import ConfigWizard

EXAMPLE = load_config(
    Path(__file__).resolve().parents[1] / "configurations.toml.example"
)
OPENAI_BASE = ConfigWizard.API_BASES["OpenAI"]
OPENROUTER_BASE = ConfigWizard.API_BASES["OpenRouter"]
OPENAI = {"engine": "gpt-4o-mini", "api_base": OPENAI_BASE}
ENTER = object()


def make_wizard(**overrides):
    wizard = ConfigWizard(config=apply_config_overrides(overrides, EXAMPLE))
    wizard.console = Console(file=io.StringIO())
    return wizard


@contextmanager
def answering(*answers):
    """
    Patch every prompt to give the scripted answers in order; ENTER and unscripted
    prompts take the default. Yields the ask mock, and fails if an answer is unused.
    """
    queue = list(answers)

    def ask(*args, default=None, **kwargs):
        answer = queue.pop(0) if queue else ENTER
        return default if answer is ENTER else answer

    ask_mock = Mock(side_effect=ask)
    prompt = Mock(ask=ask_mock)
    with patch.multiple(
        "autoresttest.tui.config_wizard",
        Prompt=prompt,
        IntPrompt=prompt,
        FloatPrompt=prompt,
        Confirm=prompt,
    ):
        yield ask_mock
    if queue:
        raise AssertionError(f"Scripted answers were not used: {queue}")


def provider_default(ask_mock):
    """The 1-based option the provider prompt offered as its default."""
    call = next(
        call
        for call in ask_mock.call_args_list
        if "Select LLM provider" in call.args[0]
    )
    return call.kwargs["default"]


class LlmDefaultsTests(unittest.TestCase):
    def test_enter_keeps_an_openai_engine(self):
        # The provider prompt used to default to OpenRouter and its first model.
        with answering():
            self.assertEqual(make_wizard(llm=OPENAI)._configure_llm(), {})

    def test_enter_keeps_a_listed_openrouter_model(self):
        llm = {"engine": "google/gemini-3-flash-preview", "api_base": OPENROUTER_BASE}
        with answering():
            self.assertEqual(make_wizard(llm=llm)._configure_llm(), {})

    def test_enter_keeps_an_unlisted_model_of_the_configured_provider(self):
        llm = {"engine": "google/gemma-4-26b-a4b-it", "api_base": OPENROUTER_BASE}
        with answering():
            self.assertEqual(make_wizard(llm=llm)._configure_llm(), {})

    def test_local_api_base_selects_the_local_provider(self):
        llm = {"engine": "local-model", "api_base": ConfigWizard.API_BASES["Local"]}
        with answering() as ask:
            self.assertEqual(make_wizard(llm=llm)._configure_llm(), {})
        self.assertEqual(provider_default(ask), "3")

    def test_unknown_api_base_selects_custom_and_keeps_it(self):
        for api_base in ("http://127.0.0.1:8080/v1", OPENAI_BASE + "/"):
            with self.subTest(api_base=api_base):
                llm = {"engine": "gemma", "api_base": api_base}
                with answering() as ask:
                    self.assertEqual(make_wizard(llm=llm)._configure_llm(), {})
                self.assertEqual(provider_default(ask), "4")

    def test_choosing_another_provider_uses_its_first_model(self):
        with answering("2"):  # OpenRouter
            overrides = make_wizard(llm=OPENAI)._configure_llm()
        self.assertEqual(
            overrides["llm"],
            {
                "engine": ConfigWizard.LLM_ENGINES["OpenRouter"][0][0],
                "api_base": OPENROUTER_BASE,
            },
        )

    def test_choosing_a_provider_that_lists_the_engine_keeps_the_engine(self):
        llm = {"engine": "gpt-5-nano-2025-08-07", "api_base": OPENROUTER_BASE}
        with answering("1"):  # OpenAI
            overrides = make_wizard(llm=llm)._configure_llm()
        self.assertEqual(overrides, {"llm": {"api_base": OPENAI_BASE}})

    def test_creative_temperature_leaves_strict_temperature_alone(self):
        # Provider, model (custom, as gpt-4o-mini is not listed) and model ID, then 0.5.
        with answering(ENTER, ENTER, ENTER, 0.5):
            overrides = make_wizard(llm=OPENAI)._configure_llm()
        self.assertEqual(overrides, {"llm": {"creative_temperature": 0.5}})


class RequestGenerationDefaultsTests(unittest.TestCase):
    def test_enter_keeps_a_duration_without_a_preset(self):
        wizard = make_wizard(request_generation={"time_duration": 900})
        with answering():
            self.assertEqual(wizard._configure_request_generation(), {})

    def test_enter_keeps_a_preset_duration(self):
        wizard = make_wizard(request_generation={"time_duration": 1800})
        with answering():
            self.assertEqual(wizard._configure_request_generation(), {})

    def test_custom_duration_can_be_changed(self):
        wizard = make_wizard(request_generation={"time_duration": 900})
        with answering(ENTER, 1500):  # the custom entry, then a new value
            overrides = wizard._configure_request_generation()
        self.assertEqual(overrides, {"request_generation": {"time_duration": 1500}})


class ApiAndAgentTests(unittest.TestCase):
    def test_override_url_can_be_turned_off(self):
        with answering(False) as ask:
            overrides = make_wizard(api={"override_url": True})._configure_api()
        self.assertEqual(overrides, {"api": {"override_url": False}})
        self.assertEqual(ask.call_count, 1)  # no host or port prompt

    def test_override_url_can_be_turned_on(self):
        with answering(True, "api.example.test", 9000):
            overrides = make_wizard(api={"override_url": False})._configure_api()
        self.assertEqual(
            overrides,
            {"api": {"override_url": True, "host": "api.example.test", "port": 9000}},
        )

    def test_enter_keeps_override_url_settings(self):
        with answering():
            self.assertEqual(
                make_wizard(api={"override_url": True})._configure_api(), {}
            )

    def test_parallel_value_generation_can_be_turned_off(self):
        wizard = make_wizard(agent={"value": {"parallelize": True, "max_workers": 4}})
        with answering(ENTER, False) as ask:  # keep the header agent setting, then No
            overrides = wizard._configure_agents()
        self.assertEqual(
            overrides, {"agent": {"value": {"parallelize": False, "max_workers": 4}}}
        )
        self.assertEqual(ask.call_count, 2)  # no worker count prompt

    def test_enter_keeps_agent_settings(self):
        wizard = make_wizard(agent={"value": {"parallelize": True, "max_workers": 4}})
        with answering():
            self.assertEqual(wizard._configure_agents(), {})


class FullSetupTests(unittest.TestCase):
    def test_enter_at_every_prompt_changes_nothing(self):
        wizard = make_wizard(
            llm=OPENAI,
            request_generation={"time_duration": 900},
            api={"override_url": True},
        )
        with (
            patch.object(wizard, "_find_spec_files", return_value=[]),
            answering("2"),  # Full Setup, then Enter everywhere
        ):
            self.assertEqual(wizard.run(), {})


if __name__ == "__main__":
    unittest.main()
