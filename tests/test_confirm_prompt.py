import unittest
from unittest.mock import patch

from autoresttest.tui import TUIDisplay


class ConfirmPromptTests(unittest.TestCase):
    def confirm(self, side_effect):
        with patch("builtins.input", side_effect=side_effect):
            return TUIDisplay().confirm("Start testing?")

    def test_ctrl_c_cancels(self):
        self.assertFalse(self.confirm(KeyboardInterrupt))

    def test_missing_input_uses_the_default(self):
        self.assertTrue(self.confirm(EOFError))

    def test_answers(self):
        self.assertTrue(self.confirm([""]))
        self.assertTrue(self.confirm(["yes"]))
        self.assertFalse(self.confirm(["n"]))


if __name__ == "__main__":
    unittest.main()
