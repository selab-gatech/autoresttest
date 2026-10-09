import shelve
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from autoresttest import autoresttest as app
from autoresttest.config import apply_config_overrides, load_config
from autoresttest.llm import LanguageModel

CONFIG = apply_config_overrides(
    {"cache": {"use_cached_table": True}},
    load_config(Path(__file__).resolve().parents[1] / "configurations.toml.example"),
)


class QTableCacheTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.object(LanguageModel, "failures", Counter()),
            patch.object(LanguageModel, "successful_queries", 0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.shelf = Path(directory.name) / "q_tables"

    def run_setup(self, failures=0, successes=0):
        """Run Q-table setup, where value-table generation makes the given LLM queries."""
        learner = MagicMock()
        learner.header_agent.q_table = {}

        def generate_value_tables(progress_callback=None):
            if failures:
                LanguageModel.failures["APIConnectionError"] += failures
            LanguageModel.successful_queries += successes
            learner.value_agent.q_table = {"listItems": {"params": {}, "body": {}}}

        learner.value_agent.initialize_q_table.side_effect = generate_value_tables
        tui = Mock()
        with (
            patch.object(app, "construct_db_dir"),
            patch.object(app, "QLearning", return_value=learner),
            patch.object(app, "get_q_table_cache_path", return_value=self.shelf),
            patch.object(app, "output_q_table"),
            patch.object(app, "InitializationProgressDisplay", MagicMock()),
        ):
            app.AutoRestTest("unused", CONFIG, tui).perform_q_learning(
                Mock(operation_nodes={"listItems": Mock()}), "probe"
            )
        with shelve.open(str(self.shelf)) as db:
            cached = "probe" in db
        messages = [call.args[0] for call in tui.print_step.call_args_list]
        return cached, messages

    def test_tables_are_not_cached_when_every_llm_query_failed(self):
        cached, messages = self.run_setup(failures=12)
        self.assertFalse(cached)
        self.assertIn(
            "Q-tables not cached: no LLM query succeeded, so the next run regenerates them",
            messages,
        )

    def test_tables_are_cached_when_some_llm_queries_succeeded(self):
        cached, _ = self.run_setup(failures=3, successes=9)
        self.assertTrue(cached)

    def test_tables_are_cached_when_no_llm_query_was_needed(self):
        cached, _ = self.run_setup()
        self.assertTrue(cached)


class SuccessfulQueryCountTests(unittest.TestCase):
    def test_successful_queries_are_counted(self):
        with (
            patch.object(LanguageModel, "cache", {}),
            patch.object(LanguageModel, "successful_queries", 0),
            patch("autoresttest.llm.llm.OpenAI") as client,
        ):
            client.return_value.chat.completions.create.return_value = Mock(
                usage=None, choices=[Mock(message=Mock(content="{}"))]
            )
            model = LanguageModel(config=CONFIG.model_copy(
                update={"llm": CONFIG.llm.model_copy(update={"api_base": "http://127.0.0.1:9/v1"})}
            ))
            model.query("prompt")
            self.assertEqual(LanguageModel.successful_queries, 1)


if __name__ == "__main__":
    unittest.main()
