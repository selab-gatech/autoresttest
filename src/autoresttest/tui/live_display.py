"""Live display for Q-learning execution phase."""

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .themes import DEFAULT_THEME, TUITheme

# Seconds between plain status lines when output is not a terminal (e.g. Docker logs),
# where the live display cannot redraw in place.
STATUS_INTERVAL_SECONDS = 60


class LiveDisplay:
    """Real-time live display for Q-learning request generation phase."""

    # Maximum individual status codes to show before grouping
    MAX_INDIVIDUAL_CODES = 8

    def __init__(
        self,
        time_duration: int,
        total_operations: int,
        theme: TUITheme = DEFAULT_THEME,
        width: int = 100,
    ):
        self.console = Console(width=width)
        self.theme = theme
        self.width = width
        self.time_duration = time_duration
        self.total_operations = total_operations

        # State tracking
        self.start_time: float = 0.0
        self.current_operation: str = ""
        self.responses: Counter = Counter()
        self.unique_errors: int = 0
        self.successful_operations: Set[str] = set()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.mutation_count: int = 0
        self.dependencies_discovered: int = 0
        self._last_status_time: float = 0.0

        # Live display instance
        self._live: Optional[Live] = None

    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_time_verbose(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _create_time_panel(self) -> Panel:
        """Create the time elapsed/remaining panel with enhanced visuals."""
        elapsed = time.time() - self.start_time
        remaining = max(0, self.time_duration - elapsed)
        progress_pct = min(100, (elapsed / self.time_duration) * 100)
        remaining_pct = 100 - progress_pct

        # Calculate progress bar dimensions
        bar_width = min(self.width - 30, 70)
        filled = int((progress_pct / 100) * bar_width)
        empty = bar_width - filled

        # Create animated progress bar with gradient effect (based on % complete)
        if progress_pct < 25:
            fill_char = "━"
            fill_color = self.theme.info
        elif progress_pct < 50:
            fill_char = "━"
            fill_color = self.theme.primary
        elif progress_pct < 75:
            fill_char = "━"
            fill_color = self.theme.warning
        else:
            fill_char = "━"
            fill_color = self.theme.success

        # Time display with clear labels
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)

        # Remaining time color based on percentage remaining (not absolute time)
        # Red: < 5% remaining, Yellow: < 20% remaining, Normal: >= 20%
        remaining_color = self.theme.error if remaining_pct < 5 else (
            self.theme.warning if remaining_pct < 20 else self.theme.text
        )

        # Build time display
        time_line = Text()
        time_line.append("  Time Elapsed: ", style=self.theme.text_dim)
        time_line.append(elapsed_str, style=f"bold {self.theme.primary}")
        time_line.append("  │  ", style=self.theme.text_dim)
        time_line.append("Time Remaining: ", style=self.theme.text_dim)
        time_line.append(remaining_str, style=f"bold {remaining_color}")

        # Build progress bar using Text objects with explicit styles (not markup strings)
        progress_line = Text()
        progress_line.append("  ", style="")
        if filled > 0 and empty > 0:
            progress_line.append(fill_char * (filled - 1), style=fill_color)
            progress_line.append("▶", style=f"bold {fill_color}")
            progress_line.append("─" * empty, style=self.theme.progress_remaining)
        elif filled > 0:
            progress_line.append(fill_char * filled, style=fill_color)
        else:
            progress_line.append("─" * bar_width, style=self.theme.progress_remaining)
        progress_line.append(" ", style="")
        progress_line.append(f"{progress_pct:5.1f}%", style=f"bold {fill_color}")

        return Panel(
            Group(
                Align.center(time_line),
                Text(),
                Align.center(progress_line),
            ),
            box=ROUNDED,
            border_style=self.theme.accent,
            title="[bold]Progress[/bold]",
            title_align="left",
            padding=(0, 1),
        )

    def _group_status_codes(self) -> List[Tuple[str, int, str]]:
        """Group status codes by category when there are many.

        Returns list of (label, count, color) tuples.
        """
        if len(self.responses) <= self.MAX_INDIVIDUAL_CODES:
            # Show individual codes
            result = []
            for code in sorted(self.responses.keys()):
                count = self.responses[code]
                color = self.theme.get_status_color(code)
                result.append((str(code), count, color))
            return result

        # Group by category
        categories = {
            "2xx Success": (0, self.theme.status_2xx),
            "3xx Redirect": (0, self.theme.status_3xx),
            "4xx Client Error": (0, self.theme.status_4xx),
            "5xx Server Error": (0, self.theme.status_5xx),
            "Other": (0, self.theme.text_dim),
        }

        for code, count in self.responses.items():
            if 200 <= code < 300:
                categories["2xx Success"] = (categories["2xx Success"][0] + count, categories["2xx Success"][1])
            elif 300 <= code < 400:
                categories["3xx Redirect"] = (categories["3xx Redirect"][0] + count, categories["3xx Redirect"][1])
            elif 400 <= code < 500:
                categories["4xx Client Error"] = (categories["4xx Client Error"][0] + count, categories["4xx Client Error"][1])
            elif 500 <= code < 600:
                categories["5xx Server Error"] = (categories["5xx Server Error"][0] + count, categories["5xx Server Error"][1])
            else:
                categories["Other"] = (categories["Other"][0] + count, categories["Other"][1])

        # Return non-zero categories
        result = []
        for label, (count, color) in categories.items():
            if count > 0:
                result.append((label, count, color))
        return result

    def _create_status_panel(self) -> Panel:
        """Create the status code distribution panel with smart grouping."""
        status_data = self._group_status_codes()

        if not status_data:
            return Panel(
                Align.center(Text("No requests sent yet...", style=self.theme.text_dim)),
                title="[bold]Status Code Distribution[/bold]",
                title_align="left",
                box=ROUNDED,
                border_style=self.theme.secondary,
                padding=(0, 1),
            )

        status_table = Table(
            box=None,
            show_header=True,
            header_style=f"bold {self.theme.text}",
            padding=(0, 1),
            expand=True,
        )
        status_table.add_column("Status", width=18, justify="left")
        status_table.add_column("Distribution", min_width=30)
        status_table.add_column("Count", width=12, justify="right")

        max_count = max(count for _, count, _ in status_data) if status_data else 1
        total_requests = sum(count for _, count, _ in status_data)

        for label, count, color in status_data:
            # Calculate bar width proportionally
            bar_width = min(40, max(1, int((count / max_count) * 40)))
            pct = (count / total_requests) * 100 if total_requests > 0 else 0

            # Create visual bar
            bar = f"[{color}]{'█' * bar_width}{'░' * (40 - bar_width)}[/{color}]"

            status_table.add_row(
                f"[{color}]{label}[/{color}]",
                bar,
                f"[{color}]{count:,}[/{color}] [dim]({pct:.1f}%)[/dim]",
            )

        return Panel(
            status_table,
            title="[bold]Status Code Distribution[/bold]",
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.secondary,
            padding=(0, 1),
        )

    def _create_stats_panel(self) -> Panel:
        """Create the statistics panel."""
        stats_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
        stats_table.add_column("Metric", style=self.theme.text_dim, ratio=2)
        stats_table.add_column("Value", style=f"bold {self.theme.text}", justify="right", ratio=1)

        # Calculate statistics
        total_requests = sum(self.responses.values())
        success_pct = (len(self.successful_operations) / max(self.total_operations, 1)) * 100
        success_color = (
            self.theme.success if success_pct >= 70 else
            self.theme.warning if success_pct >= 40 else
            self.theme.error
        )

        # Requests per second
        elapsed = time.time() - self.start_time
        rps = total_requests / max(elapsed, 1)

        stats_table.add_row(
            "Successfully Processed (2xx) Operations:",
            f"[{success_color}]{len(self.successful_operations):,} / {self.total_operations}[/{success_color}]",
        )
        stats_table.add_row(
            "Operation Coverage:",
            f"[{success_color}]{success_pct:.1f}%[/{success_color}]",
        )
        stats_table.add_row(
            "Unique Server Errors (5xx):",
            f"[{self.theme.error if self.unique_errors > 0 else self.theme.success}]{self.unique_errors}[/]",
        )
        stats_table.add_row(
            "Total Requests Sent:",
            f"[{self.theme.info}]{total_requests:,}[/{self.theme.info}]",
        )
        stats_table.add_row(
            "Requests/Second:",
            f"[{self.theme.info}]{rps:.1f}[/{self.theme.info}]",
        )
        stats_table.add_row(
            "Mutations Applied:",
            f"[{self.theme.warning}]{self.mutation_count:,}[/{self.theme.warning}]",
        )
        stats_table.add_row(
            "Dependencies Discovered:",
            f"[{self.theme.accent}]{self.dependencies_discovered:,}[/{self.theme.accent}]",
        )

        return Panel(
            stats_table,
            title="[bold]Statistics[/bold]",
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.info,
            padding=(0, 1),
        )

    def _create_operation_panel(self) -> Panel:
        """Create the current operation panel."""
        op_text = Text()
        op_text.append("▶ ", style=f"bold {self.theme.success}")
        op_text.append(self.current_operation or "Initializing...", style=f"bold {self.theme.primary}")

        return Panel(
            Align.center(op_text),
            title="[bold]Current Operation[/bold]",
            title_align="left",
            box=ROUNDED,
            border_style=self.theme.accent,
            padding=(0, 1),
        )

    def _generate_display(self) -> Panel:
        """Generate the complete live display."""
        # Header
        header = Text()
        header.append("⚡ ", style=f"bold {self.theme.warning}")
        header.append("REQUEST GENERATION", style=f"bold {self.theme.primary}")
        header.append(" ⚡", style=f"bold {self.theme.warning}")

        # Combine all panels
        content = Group(
            self._create_time_panel(),
            Text(),
            self._create_stats_panel(),
            Text(),
            self._create_status_panel(),
            Text(),
            self._create_operation_panel(),
        )

        return Panel(
            content,
            title=header,
            title_align="center",
            box=DOUBLE,
            border_style=self.theme.primary,
            padding=(1, 2),
        )

    def _print_status(self):
        """Print a one-line status for non-terminal output."""
        now = time.time()
        self._last_status_time = now
        elapsed = now - self.start_time
        remaining = max(0, self.time_duration - elapsed)
        codes = ", ".join(
            f"{code}: {count}" for code, count in sorted(self.responses.items())
        )
        print(
            f"[{self._format_time(elapsed)}] Request generation: "
            f"{sum(self.responses.values())} requests, "
            f"{len(self.successful_operations)}/{self.total_operations} operations with 2xx, "
            f"{self.unique_errors} unique server errors, "
            f"{self._format_time(remaining)} remaining | status codes: {codes or 'none'}",
            flush=True,
        )

    def _refresh(self):
        """Redraw the live display, or print a periodic status line without a terminal."""
        if self._live:
            self._live.update(self._generate_display())
        elif (
            self.start_time
            and time.time() - self._last_status_time >= STATUS_INTERVAL_SECONDS
        ):
            self._print_status()

    def start(self):
        """Start the live display (terminals only; otherwise status lines are printed)."""
        self.start_time = time.time()
        self._last_status_time = self.start_time
        if not self.console.is_terminal:
            return
        self._live = Live(
            self._generate_display(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self):
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(
        self,
        current_operation: str,
        responses: Counter,
        unique_errors: int,
        successful_operations: Set[str],
        input_tokens: int = 0,
        output_tokens: int = 0,
        mutation_count: int = 0,
        dependencies_discovered: int = 0,
    ):
        """Update the live display with new data."""
        self.current_operation = current_operation
        self.responses = responses
        self.unique_errors = unique_errors
        self.successful_operations = successful_operations
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.mutation_count = mutation_count
        self.dependencies_discovered = dependencies_discovered
        self._refresh()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class ProgressDisplay:
    """Progress display for graph construction and Q-table initialization."""

    def __init__(self, theme: TUITheme = DEFAULT_THEME, width: int = 100):
        self.console = Console(width=width)
        self.theme = theme
        self.width = width

    def create_progress_bar(
        self,
        description: str,
        total: int,
        transient: bool = False,
    ) -> Progress:
        """Create a styled progress bar."""
        progress = Progress(
            SpinnerColumn(style=self.theme.primary),
            TextColumn(f"[{self.theme.text}]{description}[/{self.theme.text}]"),
            BarColumn(
                bar_width=40,
                complete_style=self.theme.progress_complete,
                finished_style=self.theme.success,
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(f"[{self.theme.text_dim}]•[/{self.theme.text_dim}]"),
            TextColumn(f"[{self.theme.info}]{{task.completed}}/{{task.total}}[/{self.theme.info}]"),
            console=self.console,
            transient=transient,
        )
        return progress

    def track_operation(self, description: str, total: int):
        """Create a tracking context for operations."""
        return self.create_progress_bar(description, total).__enter__()


class InitializationProgressDisplay:
    """Live progress display for Q-table initialization (Value Agent, Header Agent)."""

    def __init__(
        self,
        title: str,
        total_operations: int,
        theme: TUITheme = DEFAULT_THEME,
        width: int = 100,
    ):
        self.console = Console(width=width)
        self.theme = theme
        self.width = width
        self.title = title
        self.total_operations = total_operations

        # State tracking
        self.completed_operations: int = 0
        self.current_operation: str = ""
        self.start_time: float = 0.0
        self._last_status_time: float = 0.0

        # Live display instance
        self._live: Optional[Live] = None

    def _format_time(self, seconds: float) -> str:
        """Format seconds into MM:SS."""
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes:02d}:{secs:02d}"

    def _generate_display(self) -> Panel:
        """Generate the progress display panel."""
        elapsed = time.time() - self.start_time
        progress_pct = (self.completed_operations / max(self.total_operations, 1)) * 100

        # Progress bar
        bar_width = min(self.width - 30, 60)
        filled = int((progress_pct / 100) * bar_width)
        empty = bar_width - filled

        bar_color = self.theme.primary if progress_pct < 100 else self.theme.success

        # Build display content
        content_lines = []

        # Progress line - build using Text with explicit styles (not markup strings)
        progress_text = Text()
        progress_text.append("  ", style="")
        if filled > 0 and empty > 0:
            progress_text.append("━" * filled, style=bar_color)
            progress_text.append("─" * empty, style=self.theme.progress_remaining)
        elif filled > 0:
            progress_text.append("━" * filled, style=bar_color)
        else:
            progress_text.append("─" * bar_width, style=self.theme.progress_remaining)
        progress_text.append(" ", style="")
        progress_text.append(f"{progress_pct:5.1f}%", style=f"bold {bar_color}")
        content_lines.append(Align.center(progress_text))
        content_lines.append(Text())

        # Stats line
        stats_text = Text()
        stats_text.append("  Operations: ", style=self.theme.text_dim)
        stats_text.append(
            f"{self.completed_operations}/{self.total_operations}",
            style=f"bold {self.theme.info}",
        )
        stats_text.append("  │  ", style=self.theme.text_dim)
        stats_text.append("Elapsed: ", style=self.theme.text_dim)
        stats_text.append(self._format_time(elapsed), style=f"bold {self.theme.primary}")
        content_lines.append(Align.center(stats_text))
        content_lines.append(Text())

        # Current operation
        if self.current_operation:
            op_text = Text()
            op_text.append("  ▶ ", style=f"bold {self.theme.success}")
            # Truncate long operation names
            op_name = self.current_operation
            max_op_len = self.width - 20
            if len(op_name) > max_op_len:
                op_name = op_name[: max_op_len - 3] + "..."
            op_text.append(op_name, style=f"{self.theme.text}")
            content_lines.append(Align.center(op_text))

        # Header
        header = Text()
        header.append("⚙ ", style=f"bold {self.theme.warning}")
        header.append(self.title, style=f"bold {self.theme.primary}")

        return Panel(
            Group(*content_lines),
            title=header,
            title_align="center",
            box=ROUNDED,
            border_style=self.theme.accent,
            padding=(1, 2),
        )

    def _print_status(self):
        """Print a one-line status for non-terminal output."""
        now = time.time()
        self._last_status_time = now
        print(
            f"[{self._format_time(now - self.start_time)}] {self.title}: "
            f"{self.completed_operations}/{self.total_operations} operations",
            flush=True,
        )

    def _refresh(self):
        """Redraw the live display, or print a periodic status line without a terminal."""
        if self._live:
            self._live.update(self._generate_display())
        elif (
            self.start_time
            and time.time() - self._last_status_time >= STATUS_INTERVAL_SECONDS
        ):
            self._print_status()

    def start(self):
        """Start the live progress display (terminals only; otherwise status lines are printed)."""
        self.start_time = time.time()
        self._last_status_time = self.start_time
        if not self.console.is_terminal:
            return
        self._live = Live(
            self._generate_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def stop(self):
        """Stop the live progress display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(self, current_operation: str, completed: int):
        """Update progress with current operation and completion count."""
        self.current_operation = current_operation
        self.completed_operations = completed
        self._refresh()

    def increment(self, operation_name: str = ""):
        """Increment completed count and optionally update current operation."""
        self.completed_operations += 1
        if operation_name:
            self.current_operation = operation_name
        self._refresh()

    def set_current(self, operation_name: str):
        """Set the current operation being processed."""
        self.current_operation = operation_name
        self._refresh()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
