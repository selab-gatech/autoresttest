"""Interactive configuration wizard for AutoRestTest."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.align import Align
from rich.box import DOUBLE, ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from autoresttest.config import Config, get_config

# Re-export for callers that previously imported this helper from the wizard.
from autoresttest.config import apply_config_overrides as apply_config_overrides

from .themes import DEFAULT_THEME, TUITheme


class ConfigWizard:
    """Interactive configuration wizard with beautiful TUI."""

    # Popular LLM engines organized by provider
    LLM_ENGINES = {
        "OpenAI": [
            (
                "gpt-5-mini-2025-08-07",
                "GPT-5 Mini - Recommended balance of speed & quality",
            ),
            (
                "gpt-5-nano-2025-08-07",
                "GPT-5 Nano - Budget-friendly, may be less accurate",
            ),
            ("gpt-5.2-2025-12-11", "GPT-5.2 - Higher intelligence, slower"),
        ],
        "OpenRouter": [
            (
                "google/gemini-2.5-flash-lite-preview-09-2025",
                "Gemini 2.5 Flash Lite - Recommended: cheap & fast",
            ),
            (
                "google/gemini-3-flash-preview",
                "Gemini 3 Flash - Fast with higher intelligence",
            ),
            ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5 - Fast & smart"),
            (
                "meta-llama/llama-3.3-70b-instruct",
                "Llama 3.3 70B - Efficient open-source",
            ),
        ],
        "Local": [
            ("local-model", "Local Model - Custom endpoint"),
        ],
    }

    API_BASES = {
        "OpenAI": "https://api.openai.com/v1",
        "OpenRouter": "https://openrouter.ai/api/v1",
        "Local": "http://localhost:1234/v1",
    }

    def __init__(
        self,
        theme: TUITheme = DEFAULT_THEME,
        width: int = 100,
        config: Config | None = None,
    ):
        self.console = Console(force_terminal=True, width=width)
        self.theme = theme
        self.width = width
        self._default_config = config if config is not None else get_config()

    def _print_section(self, title: str, icon: str = ""):
        """Print a styled section header."""
        header = Text()
        if icon:
            header.append(f"{icon} ", style=self.theme.primary)
        header.append(title, style=f"bold {self.theme.primary}")

        panel = Panel(
            Align.center(header),
            box=ROUNDED,
            border_style=self.theme.accent,
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)

    def _print_option_table(
        self, options: List[Tuple[str, str]], title: str = "Options"
    ):
        """Print a table of numbered options."""
        table = Table(
            box=ROUNDED,
            border_style=self.theme.accent,
            show_header=True,
            header_style=f"bold {self.theme.secondary}",
            padding=(0, 1),
        )
        table.add_column(
            "#", style=f"bold {self.theme.primary}", width=4, justify="center"
        )
        table.add_column("Option", style=self.theme.text)
        table.add_column("Description", style=self.theme.text_dim)

        for i, (option, desc) in enumerate(options, 1):
            table.add_row(str(i), option, desc)

        self.console.print(table)

    def _select_option(
        self,
        options: List[Tuple[str, str]],
        prompt_text: str,
        default_index: int = 0,
    ) -> int:
        """Display options and get user selection."""
        self._print_option_table(options)

        while True:
            self.console.print()
            default_display = f"[{self.theme.text_dim}]default: {default_index + 1}[/{self.theme.text_dim}]"
            arrow = f"[{self.theme.symbol_arrow_color}]{self.theme.symbol_arrow}[/{self.theme.symbol_arrow_color}]"
            response = Prompt.ask(
                f"  {arrow} {prompt_text} {default_display}",
                default=str(default_index + 1),
                console=self.console,
            )

            try:
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    return idx
                self.console.print(
                    f"  [red]Please enter a number between 1 and {len(options)}[/red]"
                )
            except ValueError:
                if response.strip() == "":
                    return default_index
                self.console.print("  [red]Please enter a valid number[/red]")

    def _prompt_value(
        self,
        prompt_text: str,
        default: Any,
        value_type: type = str,
        validation: Optional[callable] = None,
    ) -> Any:
        """Prompt for a value with type conversion and optional validation."""
        default_display = (
            f"[{self.theme.text_dim}]default: {default}[/{self.theme.text_dim}]"
        )
        arrow = f"[{self.theme.symbol_arrow_color}]{self.theme.symbol_arrow}[/{self.theme.symbol_arrow_color}]"

        while True:
            if value_type == bool:
                return Confirm.ask(
                    f"  {arrow} {prompt_text}",
                    default=default,
                    console=self.console,
                )
            elif value_type == int:
                result = IntPrompt.ask(
                    f"  {arrow} {prompt_text} {default_display}",
                    default=default,
                    console=self.console,
                )
            elif value_type == float:
                result = FloatPrompt.ask(
                    f"  {arrow} {prompt_text} {default_display}",
                    default=default,
                    console=self.console,
                )
            else:
                result = Prompt.ask(
                    f"  {arrow} {prompt_text} {default_display}",
                    default=str(default),
                    console=self.console,
                )

            if validation:
                valid, error_msg = validation(result)
                if not valid:
                    self.console.print(f"  [red]{error_msg}[/red]")
                    continue

            return result

    def _find_spec_files(self) -> List[Tuple[str, str]]:
        """Find OpenAPI specification files in the project."""
        spec_dirs = [
            Path("specs"),
            Path("aratrl-openapi"),
        ]

        specs = []
        for spec_dir in spec_dirs:
            if spec_dir.exists():
                for ext in ["*.yaml", "*.yml", "*.json"]:
                    for spec_file in spec_dir.rglob(ext):
                        rel_path = str(spec_file)
                        specs.append((rel_path, f"Found in {spec_dir}"))

        # Limit to first 8 for display (has 2 other defaults, so 10 total)
        if len(specs) > 8:
            specs = specs[:8]
            specs.append(("... more available", "Enter custom path"))

        return specs

    def run(self, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
        """Run the configuration wizard.

        Args:
            quick_mode: If True, only prompt for essential settings

        Returns:
            Dictionary of configuration overrides, or None if cancelled
        """
        self.console.clear()

        # Welcome header
        welcome = Text()
        welcome.append("Configuration Wizard", style=f"bold {self.theme.primary}")

        desc = Text(
            "Configure AutoRestTest interactively. Press Enter to keep defaults.",
            style=self.theme.text_dim,
        )

        panel = Panel(
            Group(Align.center(welcome), Text(), Align.center(desc)),
            box=DOUBLE,
            border_style=self.theme.primary,
            padding=(1, 2),
        )
        self.console.print(panel)

        overrides: Dict[str, Any] = {}

        try:
            # Mode selection
            if not quick_mode:
                self._print_section("Setup Mode", "")

                modes = [
                    ("Quick Setup", "Configure essential settings only (spec, LLM)"),
                    ("Full Setup", "Configure all available settings"),
                    ("Use Defaults", "Start with current configurations.toml"),
                ]

                mode_idx = self._select_option(
                    modes, "Select setup mode", default_index=2
                )

                if mode_idx == 2:  # Use defaults
                    self.console.print(
                        f"\n  [{self.theme.symbol_success_color}]{self.theme.symbol_success}[/{self.theme.symbol_success_color}] Using default configuration"
                    )
                    return {}
                elif mode_idx == 0:
                    quick_mode = True

            # 1. Specification Selection
            overrides.update(self._configure_spec())

            # 2. LLM Configuration
            overrides.update(self._configure_llm())

            if not quick_mode:
                # 3. Q-Learning Parameters
                overrides.update(self._configure_q_learning())

                # 4. Request Generation
                overrides.update(self._configure_request_generation())

                # 5. Cache Settings
                overrides.update(self._configure_cache())

                # 6. API Override
                overrides.update(self._configure_api())

                # 7. Agent Settings
                overrides.update(self._configure_agents())

            # Summary
            self._print_config_summary(overrides)

            arrow = f"[{self.theme.symbol_arrow_color}]{self.theme.symbol_arrow}[/{self.theme.symbol_arrow_color}]"
            warning = f"[{self.theme.symbol_warning_color}]{self.theme.symbol_warning}[/{self.theme.symbol_warning_color}]"
            if Confirm.ask(
                f"\n  {arrow} Proceed with this configuration?",
                default=True,
                console=self.console,
            ):
                return overrides
            else:
                self.console.print(f"\n  {warning} Configuration cancelled")
                return None

        except KeyboardInterrupt:
            warning = f"[{self.theme.symbol_warning_color}]{self.theme.symbol_warning}[/{self.theme.symbol_warning_color}]"
            self.console.print(f"\n\n  {warning} Configuration cancelled")
            return None

    def _configure_spec(self) -> Dict[str, Any]:
        """Configure specification settings."""
        self._print_section("API Specification", "")

        overrides = {}

        # Find available specs
        specs = self._find_spec_files()

        if specs:
            specs.insert(0, ("Enter custom path", "Specify your own spec file path"))

            self.console.print(
                f"\n  [dim]Current: {self._default_config.specification_location}[/dim]"
            )
            idx = self._select_option(specs, "Select specification", default_index=0)

            if idx == 0:  # Custom path
                spec_path = self._prompt_value(
                    "Enter specification path (relative to project root)",
                    self._default_config.specification_location,
                )
            else:
                spec_path = specs[idx][0]

            if spec_path != self._default_config.specification_location:
                overrides["spec"] = {"location": spec_path}
        else:
            spec_path = self._prompt_value(
                "Enter specification path",
                self._default_config.specification_location,
            )
            if spec_path != self._default_config.specification_location:
                overrides["spec"] = {"location": spec_path}

        return overrides

    def _configure_llm(self) -> Dict[str, Any]:
        """Configure LLM settings."""
        self._print_section("LLM Configuration", "")

        overrides = {}
        llm_config = {}

        # Provider selection
        providers = [
            ("OpenAI", "Direct OpenAI API"),
            ("OpenRouter", "Access many models via OpenRouter"),
            ("Local", "Local model (LM Studio, Ollama, etc.)"),
            ("Custom", "Enter custom engine and API base"),
        ]

        self.console.print(
            f"\n  [dim]Current engine: {self._default_config.openai_llm_engine}[/dim]"
        )
        self.console.print(
            f"  [dim]Current API base: {self._default_config.llm_api_base}[/dim]"
        )

        provider_idx = self._select_option(
            providers, "Select LLM provider", default_index=1
        )
        provider = providers[provider_idx][0]

        if provider == "Custom":
            engine = self._prompt_value(
                "Enter model name", self._default_config.openai_llm_engine
            )
            api_base = self._prompt_value(
                "Enter API base URL", self._default_config.llm_api_base
            )
        else:
            # Model selection for known providers
            if provider in self.LLM_ENGINES:
                models = list(self.LLM_ENGINES[provider])
                # Add custom model ID option for all providers
                models.append(("custom", "Enter custom model ID"))
                model_idx = self._select_option(models, "Select model", default_index=0)

                if models[model_idx][0] == "custom":
                    # User wants to enter a custom model ID
                    engine = self._prompt_value(
                        "Enter model ID", self._default_config.openai_llm_engine
                    )
                else:
                    engine = models[model_idx][0]
            else:
                engine = self._default_config.openai_llm_engine

            api_base = self.API_BASES.get(provider, self._default_config.llm_api_base)

        if engine != self._default_config.openai_llm_engine:
            llm_config["engine"] = engine
        if api_base != self._default_config.llm_api_base:
            llm_config["api_base"] = api_base

        # Temperature settings
        temp = self._prompt_value(
            "Creative temperature (0.0-2.0)",
            self._default_config.creative_temperature,
            float,
        )
        if temp != self._default_config.creative_temperature:
            llm_config["creative_temperature"] = temp
            llm_config["strict_temperature"] = temp

        # Max tokens
        max_tokens = self._prompt_value(
            "Max tokens (-1 for provider default)",
            self._default_config.llm_max_tokens,
            int,
        )
        if max_tokens != self._default_config.llm_max_tokens:
            llm_config["max_tokens"] = max_tokens

        if llm_config:
            overrides["llm"] = llm_config

        return overrides

    def _configure_q_learning(self) -> Dict[str, Any]:
        """Configure Q-learning parameters."""
        self._print_section("Q-Learning Parameters", "")

        overrides = {}
        q_config = {}

        self.console.print(
            "\n  [dim]These settings control the reinforcement learning behavior.[/dim]"
        )

        learning_rate = self._prompt_value(
            "Learning rate (alpha, 0.0-1.0)",
            self._default_config.q_learning.learning_rate,
            float,
        )
        if learning_rate != self._default_config.q_learning.learning_rate:
            q_config["learning_rate"] = learning_rate

        discount = self._prompt_value(
            "Discount factor (gamma, 0.0-1.0)",
            self._default_config.q_learning.discount_factor,
            float,
        )
        if discount != self._default_config.q_learning.discount_factor:
            q_config["discount_factor"] = discount

        exploration = self._prompt_value(
            "Initial exploration (epsilon, 0.0-1.0)",
            self._default_config.q_learning.max_exploration,
            float,
        )
        if exploration != self._default_config.q_learning.max_exploration:
            q_config["max_exploration"] = exploration

        if q_config:
            overrides["q_learning"] = q_config

        return overrides

    def _configure_request_generation(self) -> Dict[str, Any]:
        """Configure request generation settings."""
        self._print_section("Request Generation", "")

        overrides = {}
        req_config = {}

        # Time duration with helpful presets
        durations = [
            ("300", "5 minutes - Quick test"),
            ("600", "10 minutes - Short run"),
            ("1200", "20 minutes - Standard (default)"),
            ("1800", "30 minutes - Extended"),
            ("3600", "60 minutes - Long run"),
            ("custom", "Enter custom duration"),
        ]

        self.console.print(
            f"\n  [dim]Current duration: {self._default_config.request_generation.time_duration}s[/dim]"
        )
        idx = self._select_option(durations, "Select test duration", default_index=2)

        if durations[idx][0] == "custom":
            duration = self._prompt_value(
                "Enter duration in seconds",
                self._default_config.request_generation.time_duration,
                int,
            )
        else:
            duration = int(durations[idx][0])

        if duration != self._default_config.request_generation.time_duration:
            req_config["time_duration"] = duration

        mutation_rate = self._prompt_value(
            "Mutation rate (0.0-1.0)",
            self._default_config.request_generation.mutation_rate,
            float,
        )
        if mutation_rate != self._default_config.request_generation.mutation_rate:
            req_config["mutation_rate"] = mutation_rate

        if req_config:
            overrides["request_generation"] = req_config

        return overrides

    def _configure_cache(self) -> Dict[str, Any]:
        """Configure cache settings."""
        self._print_section("Cache Settings", "")

        overrides = {}
        cache_config = {}

        self.console.print(
            "\n  [dim]Caching speeds up repeated runs by reusing computed data.[/dim]"
        )

        use_graph = self._prompt_value(
            "Use cached graph?",
            self._default_config.cache.use_cached_graph,
            bool,
        )
        if use_graph != self._default_config.cache.use_cached_graph:
            cache_config["use_cached_graph"] = use_graph

        use_table = self._prompt_value(
            "Use cached Q-tables?",
            self._default_config.cache.use_cached_table,
            bool,
        )
        if use_table != self._default_config.cache.use_cached_table:
            cache_config["use_cached_table"] = use_table

        if cache_config:
            overrides["cache"] = cache_config

        return overrides

    def _configure_api(self) -> Dict[str, Any]:
        """Configure API URL override settings."""
        self._print_section("API URL Override", "")

        overrides = {}
        api_config = {}

        self.console.print(
            "\n  [dim]Override the API URL from the specification with a custom endpoint.[/dim]"
        )

        override_url = self._prompt_value(
            "Override API URL from spec?",
            self._default_config.api.override_url,
            bool,
        )

        if override_url:
            api_config["override_url"] = True

            host = self._prompt_value(
                "API Host",
                self._default_config.api.host,
            )
            if host != self._default_config.api.host:
                api_config["host"] = host

            port = self._prompt_value(
                "API Port",
                self._default_config.api.port,
                int,
            )
            if port != self._default_config.api.port:
                api_config["port"] = port

        if api_config:
            overrides["api"] = api_config

        return overrides

    def _configure_agents(self) -> Dict[str, Any]:
        """Configure agent-specific settings."""
        self._print_section("Agent Configuration", "")

        overrides = {}

        # Header agent
        self.console.print(
            "\n  [dim]Header Agent generates Basic Authentication headers.[/dim]"
        )

        enable_header = self._prompt_value(
            "Enable Header Agent?",
            self._default_config.enable_header_agent,
            bool,
        )
        if enable_header != self._default_config.enable_header_agent:
            overrides["agents"] = {"header": {"enabled": enable_header}}

        # Value agent parallelization
        self.console.print(
            "\n  [dim]Value Agent parallelization speeds up Q-table initialization.[/dim]"
        )

        parallelize = self._prompt_value(
            "Parallelize value generation?",
            self._default_config.parallelize_value_generation,
            bool,
        )

        if parallelize:
            workers = self._prompt_value(
                "Number of worker threads",
                self._default_config.value_generation_workers,
                int,
            )

            if (
                parallelize != self._default_config.parallelize_value_generation
                or workers != self._default_config.value_generation_workers
            ):
                if "agent" not in overrides:
                    overrides["agent"] = {}
                overrides["agent"]["value"] = {
                    "parallelize": parallelize,
                    "max_workers": workers,
                }

        return overrides

    def _print_config_summary(self, overrides: Dict[str, Any]):
        """Print a summary of the configuration changes."""
        self._print_section("Configuration Summary", "")

        if not overrides:
            info = f"[{self.theme.symbol_info_color}]{self.theme.symbol_info}[/{self.theme.symbol_info_color}]"
            self.console.print(f"\n  {info} No changes from default configuration")
            return

        table = Table(
            box=ROUNDED,
            border_style=self.theme.accent,
            show_header=True,
            header_style=f"bold {self.theme.secondary}",
        )
        table.add_column("Setting", style=self.theme.text_dim)
        table.add_column("New Value", style=f"bold {self.theme.success}")

        def flatten_dict(d: Dict, prefix: str = "") -> List[Tuple[str, str]]:
            items = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, key))
                else:
                    items.append((key, str(v)))
            return items

        for key, value in flatten_dict(overrides):
            table.add_row(key, value)

        self.console.print()
        self.console.print(Align.center(table))
