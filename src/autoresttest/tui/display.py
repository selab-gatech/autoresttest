"""Main TUI display components for AutoRestTest."""

import sys
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .themes import DEFAULT_THEME, TUITheme


class TUIDisplay:
    """Main TUI display handler for AutoRestTest."""

    def __init__(self, theme: TUITheme = DEFAULT_THEME, width: int = 80):
        self.console = Console(width=width)
        self.theme = theme
        self.width = width

    def clear(self):
        """Clear the terminal screen."""
        self.console.clear()

    def print_banner(self):
        """Display the AutoRestTest banner."""
        banner_text = """
   ___       __       ___         __  ______        __
  / _ |__ __/ /____  / _ \\___ ___/ /_/_  __/__ ___ / /_
 / __ / // / __/ _ \\/ , _/ -_|_-< __/ / / / -_|_-</ __/
/_/ |_\\_,_/\\__/\\___/_/|_|\\__/___/\\__/ /_/  \\__/___/\\__/
"""
        banner = Text(banner_text, style=f"bold {self.theme.primary}")

        subtitle = Text(
            "Automated REST API Testing with Multi-Agent Reinforcement Learning",
            style=f"italic {self.theme.text_dim}",
        )

        panel = Panel(
            Align.center(Group(banner, Align.center(subtitle))),
            box=DOUBLE,
            border_style=self.theme.primary,
            padding=(0, 2),
        )
        self.console.print(panel)

    def print_section_header(self, title: str, icon: str = ""):
        """Print a section header with styling."""
        header_text = f"{icon} {title}" if icon else title
        header = Text(header_text, style=f"bold {self.theme.primary}")
        panel = Panel(
            Align.center(header),
            box=ROUNDED,
            border_style=self.theme.accent,
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)

    def print_phase_start(self, phase_name: str, description: str = ""):
        """Display the start of a major phase."""
        title = Text()
        title.append("", style=f"bold {self.theme.warning}")
        title.append(f" {phase_name.upper()} ", style=f"bold {self.theme.primary}")
        title.append("", style=f"bold {self.theme.warning}")

        content = Group(Align.center(title))
        if description:
            desc = Text(description, style=self.theme.text_dim, justify="center")
            content = Group(Align.center(title), Text(), Align.center(desc))

        panel = Panel(
            content,
            box=HEAVY,
            border_style=self.theme.warning,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)

    def print_phase_complete(self, phase_name: str, details: Optional[str] = None):
        """Display phase completion."""
        title = Text()
        title.append(self.theme.symbol_success + " ", style=self.theme.symbol_success_color)
        title.append(f"{phase_name} Complete", style=f"bold {self.theme.success}")

        content = Align.center(title)
        if details:
            detail_text = Text(details, style=self.theme.text_dim, justify="center")
            content = Group(Align.center(title), Text(), Align.center(detail_text))

        panel = Panel(
            content,
            box=ROUNDED,
            border_style=self.theme.success,
            padding=(0, 2),
        )
        self.console.print(panel)

    def print_step(self, message: str, status: str = "info"):
        """Print a step message with appropriate styling."""
        symbols = {
            "info": (self.theme.symbol_info, self.theme.symbol_info_color),
            "success": (self.theme.symbol_success, self.theme.symbol_success_color),
            "warning": (self.theme.symbol_warning, self.theme.symbol_warning_color),
            "error": (self.theme.symbol_error, self.theme.symbol_error_color),
            "progress": (self.theme.symbol_progress, self.theme.symbol_progress_color),
        }
        symbol, color = symbols.get(status, (self.theme.symbol_bullet, self.theme.symbol_bullet_color))
        self.console.print(f"  [{color}]{symbol}[/{color}] {message}")

    def print_key_value(self, key: str, value: Any, indent: int = 2):
        """Print a key-value pair with styling."""
        spaces = " " * indent
        key_styled = Text(f"{key}:", style=f"bold {self.theme.text_dim}")
        value_styled = Text(f" {value}", style=self.theme.text)
        self.console.print(Text(spaces), key_styled, value_styled, end="")
        self.console.print()

    def print_config_summary(self, config: Dict[str, Any]):
        """Display a summary of configuration settings."""
        table = Table(
            box=ROUNDED,
            border_style=self.theme.accent,
            title="Configuration Summary",
            title_style=f"bold {self.theme.primary}",
            show_header=True,
            header_style=f"bold {self.theme.secondary}",
            padding=(0, 1),
        )

        table.add_column("Setting", style=self.theme.text_dim, no_wrap=True)
        table.add_column("Value", style=self.theme.text)

        for key, value in config.items():
            # Format boolean values with color
            if isinstance(value, bool):
                value_str = (
                    f"[{self.theme.success}]true[/{self.theme.success}]"
                    if value
                    else f"[{self.theme.error}]false[/{self.theme.error}]"
                )
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        self.console.print()
        self.console.print(Align.center(table))

    def print_token_usage(self, input_tokens: int, output_tokens: int):
        """Display token usage statistics."""
        table = Table(
            box=ROUNDED,
            border_style=self.theme.info,
            title="LLM Token Usage",
            title_style=f"bold {self.theme.info}",
            show_header=False,
            padding=(0, 1),
        )

        table.add_column("Metric", style=self.theme.text_dim)
        table.add_column("Value", style=f"bold {self.theme.text}", justify="right")

        table.add_row("Input Tokens", f"{input_tokens:,}")
        table.add_row("Output Tokens", f"{output_tokens:,}")
        table.add_row("Total Tokens", f"{input_tokens + output_tokens:,}")

        self.console.print()
        self.console.print(Align.center(table))

    def print_final_report(
        self,
        title: str,
        duration: int,
        total_requests: int,
        status_distribution: Dict[int, int],
        total_operations: int,
        successful_operations: int,
        unique_errors: int,
    ):
        """Display the final execution report."""
        # Header
        report_title = Text()
        report_title.append("FINAL REPORT", style=f"bold {self.theme.primary}")

        # Create status distribution table
        status_table = Table(
            box=ROUNDED,
            border_style=self.theme.accent,
            title="Status Code Distribution",
            title_style=f"bold {self.theme.secondary}",
            show_header=True,
            header_style=f"bold {self.theme.text}",
        )
        status_table.add_column("Code", style=self.theme.text_dim, width=6, justify="center")
        status_table.add_column("Count", justify="right", width=10)
        status_table.add_column("Distribution", width=40)

        max_count = max(status_distribution.values()) if status_distribution else 1
        for code, count in sorted(status_distribution.items()):
            color = self.theme.get_status_color(code)
            bar_width = int((count / max_count) * 35)
            bar = f"[{color}]{'█' * bar_width}{'░' * (35 - bar_width)}[/{color}]"
            status_table.add_row(
                f"[{color}]{code}[/{color}]",
                f"[{color}]{count:,}[/{color}]",
                bar,
            )

        # Summary table
        summary_table = Table(
            box=ROUNDED,
            border_style=self.theme.info,
            title="Summary Statistics",
            title_style=f"bold {self.theme.info}",
            show_header=False,
        )
        summary_table.add_column("Metric", style=self.theme.text_dim)
        summary_table.add_column("Value", style=f"bold {self.theme.text}", justify="right")

        success_pct = (successful_operations / max(total_operations, 1)) * 100
        success_color = (
            self.theme.success if success_pct >= 70 else
            self.theme.warning if success_pct >= 40 else
            self.theme.error
        )

        summary_table.add_row("API Title", title)
        summary_table.add_row("Duration", f"{duration}s ({duration // 60}m {duration % 60}s)")
        summary_table.add_row("Total Requests", f"{total_requests:,}")
        summary_table.add_row("Total Operations", str(total_operations))
        summary_table.add_row(
            "Successful Operations",
            f"[{success_color}]{successful_operations} ({success_pct:.1f}%)[/{success_color}]",
        )
        summary_table.add_row(
            "Unique Server Errors",
            f"[{self.theme.error if unique_errors > 0 else self.theme.success}]{unique_errors}[/]",
        )

        # Combine into final panel
        panel = Panel(
            Group(
                Align.center(report_title),
                Text(),
                Columns([summary_table, status_table], expand=True, equal=True),
            ),
            box=DOUBLE,
            border_style=self.theme.primary,
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)

    def print_error(self, message: str, details: Optional[str] = None):
        """Display an error message."""
        error_text = Text()
        error_text.append(self.theme.symbol_error + " ", style=self.theme.symbol_error_color)
        error_text.append(message, style=f"bold {self.theme.error}")

        content = error_text
        if details:
            detail_text = Text(f"\n{details}", style=self.theme.text_dim)
            content = Group(error_text, detail_text)

        panel = Panel(
            content,
            box=ROUNDED,
            border_style=self.theme.error,
            title="Error",
            title_align="left",
        )
        self.console.print(panel)

    def print_warning(self, message: str):
        """Display a warning message."""
        warning_text = Text()
        warning_text.append(self.theme.symbol_warning + " ", style=self.theme.symbol_warning_color)
        warning_text.append(message, style=self.theme.warning)
        self.console.print(warning_text)

    def print_success(self, message: str):
        """Display a success message."""
        success_text = Text()
        success_text.append(self.theme.symbol_success + " ", style=self.theme.symbol_success_color)
        success_text.append(message, style=self.theme.success)
        self.console.print(success_text)

    def create_progress(self) -> Progress:
        """Create a styled progress bar context manager."""
        return Progress(
            SpinnerColumn(style=self.theme.primary),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                complete_style=self.theme.progress_complete,
                finished_style=self.theme.success,
            ),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    def confirm(self, message: str, default: bool = True) -> bool:
        """Display a confirmation prompt."""
        default_str = "Y/n" if default else "y/N"
        self.console.print(f"\n[{self.theme.symbol_info_color}]{self.theme.symbol_info}[/{self.theme.symbol_info_color}] {message} [{default_str}]: ", end="")

        try:
            response = input().strip().lower()
            if not response:
                return default
            return response in ("y", "yes", "true", "1")
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return default

    def wait_for_key(self, message: str = "Press Enter to continue..."):
        """Wait for user to press Enter."""
        self.console.print(f"\n[dim]{message}[/dim]", end="")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
