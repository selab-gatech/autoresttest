import io
import unittest
from collections import Counter
from unittest.mock import patch

from rich.console import Console

from autoresttest.tui.live_display import InitializationProgressDisplay, LiveDisplay

CLOCK = "autoresttest.tui.live_display.time.time"


def with_console(display, terminal):
    display.console = Console(file=io.StringIO(), width=100, force_terminal=terminal)
    return display


def printed_lines(display, updates):
    with patch(CLOCK, return_value=1000.0):
        display.start()
    with patch("builtins.print") as mock_print:
        for now, update in updates:
            with patch(CLOCK, return_value=now):
                update()
    return [call.args[0] for call in mock_print.call_args_list]


class NonTerminalStatusTests(unittest.TestCase):
    def test_request_generation_prints_a_status_line_each_minute(self):
        display = with_console(LiveDisplay(time_duration=600, total_operations=40), False)
        update = lambda: display.update(
            "getFlightById", Counter({200: 7, 404: 3}), 1, {"getFlightById"}
        )
        lines = printed_lines(
            display, [(1030.0, update), (1061.0, update), (1090.0, update), (1122.0, update)]
        )
        self.assertIsNone(display._live)
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            lines[0],
            "[00:01:01] Request generation: 10 requests, 1/40 operations with 2xx, "
            "1 unique server errors, 00:08:59 remaining | status codes: 200: 7, 404: 3",
        )
        self.assertTrue(lines[1].startswith("[00:02:02] "))

    def test_initialization_prints_a_status_line_each_minute(self):
        display = with_console(
            InitializationProgressDisplay("Value Agent Q-Table Generation", 40), False
        )
        lines = printed_lines(
            display,
            [(1000.0 + 20 * i, lambda i=i: display.update("op", i)) for i in range(1, 7)],
        )
        self.assertIsNone(display._live)
        self.assertEqual(
            lines,
            [
                "[01:00] Value Agent Q-Table Generation: 3/40 operations",
                "[02:00] Value Agent Q-Table Generation: 6/40 operations",
            ],
        )

    def test_terminals_keep_the_live_display(self):
        display = with_console(LiveDisplay(time_duration=600, total_operations=40), True)
        lines = printed_lines(
            display, [(1120.0, lambda: display.update("op", Counter({200: 1}), 0, set()))]
        )
        try:
            self.assertIsNotNone(display._live)
            self.assertEqual(lines, [])
        finally:
            display.stop()


if __name__ == "__main__":
    unittest.main()
