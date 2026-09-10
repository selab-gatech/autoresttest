import argparse
import json
import shelve
import sys
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

from autoresttest.config import apply_config_overrides, load_config
from autoresttest.config.config import Config
from autoresttest.graph import RequestGenerator
from autoresttest.graph.generate_graph import OperationGraph
from autoresttest.llm import LanguageModel
from autoresttest.marl import QLearning
from autoresttest.models import to_dict_helper
from autoresttest.specification import SpecificationParser
from autoresttest.tui import ConfigWizard, InitializationProgressDisplay, LiveDisplay, TUIDisplay
from autoresttest.tui.themes import DEFAULT_THEME
from autoresttest.utils import (
    EmbeddingModel,
    construct_db_dir,
    get_api_url,
    get_graph_cache_path,
    get_q_table_cache_path,
    is_json_seriable,
    param_key_to_label,
)

load_dotenv()

AUTORESTTEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTORESTTEST_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def ensure_output_dir(spec_name: str) -> Path:
    output_dir = DATA_ROOT / spec_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoRestTest - Automated REST API Testing with Multi-Agent RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autoresttest                    # Run with TUI and configuration wizard
  autoresttest --quick            # Quick setup (essential settings only)
  autoresttest --skip-wizard      # Skip wizard, use configurations.toml directly

For more information, visit: https://github.com/tylerstennett/AutoRestTest
        """,
    )
    parser.add_argument(
        "--skip-wizard",
        action="store_true",
        help="Skip configuration wizard and use configurations.toml directly",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick setup wizard (essential settings only)",
    )
    parser.add_argument(
        "-s",
        "--spec",
        type=str,
        default=None,
        help="Override specification path (relative to project root)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=None,
        help="Override test duration in seconds",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="TUI display width (default: 100)",
    )
    return parser.parse_args()


def output_q_table(q_learning: QLearning, spec_name: str):
    parameter_table = q_learning.parameter_agent.q_table
    body_obj_table = q_learning.body_object_agent.q_table
    value_table = q_learning.value_agent.q_table
    operation_table = q_learning.operation_agent.q_table
    data_source_table = q_learning.data_source_agent.q_table
    dependency_table = q_learning.dependency_agent.q_table
    header_table = (
        q_learning.header_agent.q_table
        if q_learning.header_agent.q_table
        else "Disabled"
    )

    simplified_param_table = {}
    for operation, operation_values in parameter_table.items():
        simplified_param_table[operation] = {"params": {}, "body": {}}
        for parameter, parameter_values in operation_values["params"].items():
            simplified_param_table[operation]["params"][str(parameter)] = (
                parameter_values
            )
        for body, body_values in operation_values["body"].items():
            simplified_param_table[operation]["body"][str(body)] = body_values

    simplified_body_table = {}
    for operation, operation_values in body_obj_table.items():
        simplified_body_table[operation] = {}
        for mime_type, mime_values in operation_values.items():
            if mime_type not in simplified_body_table[operation]:
                simplified_body_table[operation][mime_type] = {}
            for body, body_values in mime_values.items():
                simplified_body_table[operation][mime_type][str(body)] = body_values

    compiled_q_table = {
        "OPERATION AGENT": operation_table,
        "HEADER AGENT": header_table,
        "PARAMETER AGENT": simplified_param_table,
        "VALUE AGENT": value_table,
        "BODY OBJECT AGENT": simplified_body_table,
        "DATA SOURCE AGENT": data_source_table,
        "DEPENDENCY AGENT": dependency_table,
    }
    compiled_q_table = to_dict_helper(compiled_q_table)
    output_dir = ensure_output_dir(spec_name)

    q_tables_path = output_dir / "q_tables.json"
    with q_tables_path.open("w") as f:
        json.dump(compiled_q_table, f, indent=2)


def output_successes(q_learning: QLearning, spec_name: str):
    output_dir = ensure_output_dir(spec_name)

    with (output_dir / "successful_parameters.json").open("w") as f:
        json.dump(to_dict_helper(q_learning.successful_parameters), f, indent=2)

    with (output_dir / "successful_bodies.json").open("w") as f:
        json.dump(q_learning.successful_bodies, f, indent=2)

    with (output_dir / "successful_responses.json").open("w") as f:
        json.dump(q_learning.successful_responses, f, indent=2)

    with (output_dir / "successful_primitives.json").open("w") as f:
        json.dump(q_learning.successful_primitives, f, indent=2)


def output_errors(q_learning: QLearning, spec_name: str):
    output_dir = ensure_output_dir(spec_name)

    seriable_errors = {}
    for operation_idx, unique_errors in q_learning.unique_errors.items():
        seriable_errors[operation_idx] = []
        for error in unique_errors:
            parameters = error.get("parameters")
            if isinstance(parameters, dict):
                error = {
                    **error,
                    "parameters": {
                        param_key_to_label(key) if isinstance(key, tuple) else key: value
                        for key, value in parameters.items()
                    },
                }
            if is_json_seriable(error):
                seriable_errors[operation_idx].append(error)

    with (output_dir / "server_errors.json").open("w") as f:
        json.dump(seriable_errors, f, indent=2)


def output_operation_status_codes(q_learning: QLearning, spec_name: str):
    output_dir = ensure_output_dir(spec_name)

    with (output_dir / "operation_status_codes.json").open("w") as f:
        json.dump(q_learning.operation_response_counter, f, indent=2)


def output_report(
    q_learning: QLearning, spec_name: str, spec_parser: SpecificationParser
):
    output_dir = ensure_output_dir(spec_name)

    title = spec_parser.get_api_title() if spec_parser.get_api_title() else spec_name
    title = f"'{title}' ({spec_name})"

    unique_processed_200s = set()
    for operation_idx, status_codes in q_learning.operation_response_counter.items():
        for status_code in status_codes:
            if status_code // 100 == 2:
                unique_processed_200s.add(operation_idx)

    unique_errors = 0
    for operation_idx in q_learning.unique_errors:
        unique_errors += len(q_learning.unique_errors[operation_idx])

    total_requests = sum(q_learning.responses.values())

    report_content = {
        "Title": "AutoRestTest Report for " + title,
        "Duration": f"{q_learning.time_duration} seconds",
        "Total Requests Sent": total_requests,
        "Status Code Distribution": dict(q_learning.responses),
        "Number of Total Operations": len(q_learning.operation_agent.q_table),
        "Number of Successfully Processed Operations": len(unique_processed_200s),
        "Percentage of Successfully Processed Operations": str(
            round(
                len(unique_processed_200s)
                / max(len(q_learning.operation_agent.q_table), 1)
                * 100,
                2,
            )
        )
        + "%",
        "Number of Unique Server Errors": unique_errors,
        "Operations with Server Errors": q_learning.errors,
    }

    with (output_dir / "report.json").open("w") as f:
        json.dump(report_content, f, indent=2)


def parse_specification_location(spec_loc: str):
    spec_path = Path(spec_loc).expanduser()
    return spec_path.parent, spec_path.stem, spec_path.suffix


class AutoRestTest:
    """Main AutoRestTest execution class with TUI integration."""

    def __init__(
        self,
        spec_dir: Union[Path, str],
        config: Config,
        tui: TUIDisplay,
    ):
        self.spec_dir = Path(spec_dir).expanduser()
        self.is_naive = False
        construct_db_dir()

        self.config = config
        self.tui = tui

        self.use_cached_graph = self.config.cache.use_cached_graph
        self.use_cached_table = self.config.cache.use_cached_table

    def init_graph(
        self,
        spec_name: str,
        spec_path: Union[Path, str],
        embedding_model: EmbeddingModel,
    ) -> OperationGraph:
        self.tui.print_step(f"Parsing OpenAPI specification: {spec_path}...", "progress")
        spec_parser = SpecificationParser(
            spec_path=str(spec_path), spec_name=spec_name, config=self.config
        )
        self.tui.print_step("Specification parsed successfully!", "success")

        if self.config.api.override_url:
            api_url = self.config.custom_api_url
            self.tui.print_step(f"Using custom API URL: {api_url}", "info")
        else:
            api_url = get_api_url(spec_parser)
            self.tui.print_step(f"Using API URL from specification: {api_url}", "info")

        operation_graph = OperationGraph(
            spec_path=str(spec_path),
            spec_name=spec_name,
            spec_parser=spec_parser,
            embedding_model=embedding_model,
        )
        request_generator = RequestGenerator(
            operation_graph=operation_graph, api_url=api_url, is_naive=self.is_naive
        )

        operation_graph.assign_request_generator(request_generator)
        return operation_graph

    def generate_graph(
        self, spec_name: str, ext: str, embedding_model: EmbeddingModel
    ) -> OperationGraph:
        spec_path = self.spec_dir / f"{spec_name}{ext}"
        db_graph = get_graph_cache_path(spec_name)

        self.tui.print_phase_start(
            "Semantic Operation Dependency Graph",
            "Building operation relationships and dependencies",
        )

        # Always initialize the graph first
        operation_graph = self.init_graph(spec_name, spec_path, embedding_model)

        with shelve.open(str(db_graph)) as db:
            loaded_from_shelf = False

            if spec_name in db and self.use_cached_graph:
                self.tui.print_step(f"Loading cached graph for {spec_name}...", "progress")
                try:
                    graph_properties = db[spec_name]
                    operation_graph.operation_edges = graph_properties["edges"]
                    operation_graph.operation_nodes = graph_properties["nodes"]
                    self.tui.print_step("Loaded graph from cache", "success")
                    loaded_from_shelf = True
                except Exception as e:
                    self.tui.print_step(f"Cache load failed: {e}", "warning")

            if not loaded_from_shelf:
                self.tui.print_step(f"Building new graph for {spec_name}...", "progress")
                operation_graph.create_graph()

                graph_properties = {
                    "edges": operation_graph.operation_edges,
                    "nodes": operation_graph.operation_nodes,
                }

                try:
                    db[spec_name] = graph_properties
                    self.tui.print_step("Graph cached for future runs", "success")
                except Exception as e:
                    self.tui.print_step(f"Cache save failed: {e}", "warning")

        self.tui.print_phase_complete(
            "Graph Construction",
            f"{len(operation_graph.operation_nodes)} operations discovered",
        )

        return operation_graph

    def perform_q_learning(self, operation_graph: OperationGraph, spec_name: str):
        self.tui.print_phase_start(
            "Q-Table Initialization",
            "Initializing reinforcement learning agents",
        )

        q_learning = QLearning(
            operation_graph,
            alpha=self.config.q_learning.learning_rate,
            gamma=self.config.q_learning.discount_factor,
            epsilon=self.config.q_learning.max_exploration,
            time_duration=self.config.request_generation.time_duration,
            mutation_rate=self.config.request_generation.mutation_rate,
            tui=self.tui,
        )
        db_q_table = get_q_table_cache_path(spec_name)

        # Initialize Q-tables for all agents with progress tracking
        agents = [
            ("Operation", q_learning.operation_agent),
            ("Parameter", q_learning.parameter_agent),
            ("Body Object", q_learning.body_object_agent),
            ("Dependency", q_learning.dependency_agent),
            ("Data Source", q_learning.data_source_agent),
        ]

        for agent_name, agent in agents:
            agent.initialize_q_table()
            self.tui.print_step(f"Initialized {agent_name} Agent Q-table", "success")

        output_q_table(q_learning, spec_name)

        with shelve.open(str(db_q_table)) as db:
            loaded_value_from_shelf = False
            loaded_header_from_shelf = False

            if spec_name in db and self.use_cached_table:
                self.tui.print_step(f"Loading cached Q-tables for {spec_name}...", "progress")

                compiled_q_table = db[spec_name]

                try:
                    q_learning.value_agent.q_table = compiled_q_table["value"]
                    self.tui.print_step("Loaded Value Agent Q-table from cache", "success")
                    loaded_value_from_shelf = True
                except Exception:
                    self.tui.print_step("Cache load failed for Value Agent", "warning")
                    loaded_value_from_shelf = False

                if self.config.enable_header_agent:
                    try:
                        q_learning.header_agent.q_table = compiled_q_table["header"]
                        self.tui.print_step("Loaded Header Agent Q-table from cache", "success")
                        loaded_header_from_shelf = (
                            True if q_learning.header_agent.q_table else False
                        )
                    except Exception:
                        self.tui.print_step("Cache load failed for Header Agent", "warning")
                        loaded_header_from_shelf = False

            if not loaded_value_from_shelf:
                total_ops = len(operation_graph.operation_nodes)
                with InitializationProgressDisplay(
                    title="Value Agent Q-Table Generation",
                    total_operations=total_ops,
                    width=self.tui.width,
                ) as progress:
                    def value_progress_callback(op_id: str, completed: int):
                        progress.update(op_id, completed)

                    q_learning.value_agent.initialize_q_table(
                        progress_callback=value_progress_callback
                    )

                token_counter = LanguageModel.get_tokens()
                self.tui.print_step(
                    f"Value Agent Q-table generated - Tokens: {token_counter.input_tokens:,} in / {token_counter.output_tokens:,} out",
                    "success",
                )

            if self.config.enable_header_agent and not loaded_header_from_shelf:
                total_ops = len(operation_graph.operation_nodes)
                with InitializationProgressDisplay(
                    title="Header Agent Q-Table Generation",
                    total_operations=total_ops,
                    width=self.tui.width,
                ) as progress:
                    def header_progress_callback(op_id: str, completed: int):
                        progress.update(op_id, completed)

                    q_learning.header_agent.initialize_q_table(
                        progress_callback=header_progress_callback
                    )

                token_counter = LanguageModel.get_tokens()
                self.tui.print_step(
                    f"Header Agent Q-table generated - Tokens: {token_counter.input_tokens:,} in / {token_counter.output_tokens:,} out",
                    "success",
                )
            elif not self.config.enable_header_agent:
                q_learning.header_agent.q_table = {}

            try:
                db[spec_name] = {
                    "value": q_learning.value_agent.q_table,
                    "header": q_learning.header_agent.q_table,
                }
                self.tui.print_step("Q-tables cached for future runs", "success")
            except Exception:
                self.tui.print_step("Failed to cache Q-tables", "warning")

        output_q_table(q_learning, spec_name)

        self.tui.print_phase_complete("Q-Table Initialization")
        self.tui.print_phase_start(
            "Request Generation (MARL)",
            f"Testing API for {self.config.request_generation.time_duration} seconds using Multi-Agent Reinforcement Learning",
        )

        q_learning.run()

        self.tui.print_phase_complete("Request Generation")

        return q_learning

    def print_performance(self, q_learning: QLearning, spec_parser: SpecificationParser):
        token_counter = LanguageModel.get_tokens()

        # Calculate statistics for final report
        unique_processed_200s = set()
        for operation_idx, status_codes in q_learning.operation_response_counter.items():
            for status_code in status_codes:
                if status_code // 100 == 2:
                    unique_processed_200s.add(operation_idx)

        unique_errors = sum(len(errs) for errs in q_learning.unique_errors.values())
        total_requests = sum(q_learning.responses.values())

        title = spec_parser.get_api_title() if spec_parser.get_api_title() else "API"

        self.tui.print_final_report(
            title=title,
            duration=q_learning.time_duration,
            total_requests=total_requests,
            status_distribution=dict(q_learning.responses),
            total_operations=len(q_learning.operation_agent.q_table),
            successful_operations=len(unique_processed_200s),
            unique_errors=unique_errors,
        )

        self.tui.print_token_usage(
            input_tokens=token_counter.input_tokens,
            output_tokens=token_counter.output_tokens,
        )

    def run_all(self):
        for spec_file in self.spec_dir.iterdir():
            if not spec_file.is_file():
                continue
            spec_name = spec_file.stem
            self.tui.print_section_header(f"Testing: {spec_name}")
            self.run_single(spec_name, spec_file.suffix)

    def run_single(self, spec_name: str, ext: str):
        self.tui.print_section_header(f"Testing: {spec_name}")

        embedding_model = EmbeddingModel()
        operation_graph = self.generate_graph(spec_name, ext, embedding_model)
        q_learning = self.perform_q_learning(operation_graph, spec_name)
        self.print_performance(q_learning, operation_graph.spec_parser)
        output_q_table(q_learning, spec_name)
        output_successes(q_learning, spec_name)
        output_errors(q_learning, spec_name)
        output_operation_status_codes(q_learning, spec_name)
        output_report(q_learning, spec_name, operation_graph.spec_parser)

        self.tui.print_success("AutoRestTest completed successfully!")
        self.tui.print_step(f"Results saved to: data/{spec_name}/", "info")


def main():
    args = parse_args()

    # Initialize TUI (always enabled)
    tui = TUIDisplay(width=args.width)
    tui.clear()
    tui.print_banner()

    # Resolve one configuration: file defaults, then wizard, then explicit CLI fields.
    config = load_config()
    if not args.skip_wizard:
        wizard = ConfigWizard(width=args.width, config=config)
        overrides = wizard.run(quick_mode=args.quick)

        if overrides is None:
            # User cancelled
            sys.exit(0)
        config = apply_config_overrides(overrides, config)

    # Apply CLI overrides
    cli_overrides = {}
    if args.spec is not None:
        cli_overrides["spec"] = {"location": args.spec}
    if args.time is not None:
        cli_overrides["request_generation"] = {"time_duration": args.time}
    config = apply_config_overrides(cli_overrides, config)

    # Display configuration summary
    config_summary = {
        "Specification": config.specification_location,
        "LLM Engine": config.openai_llm_engine,
        "API Base": config.llm_api_base,
        "Duration": f"{config.request_generation.time_duration}s",
        "Cache Graph": config.cache.use_cached_graph,
        "Cache Q-Tables": config.cache.use_cached_table,
    }
    tui.print_config_summary(config_summary)

    if not tui.confirm("Start testing with this configuration?"):
        tui.print_warning("Execution cancelled by user")
        sys.exit(0)

    # Parse specification location and run
    specification_directory, specification_name, ext = parse_specification_location(
        str(PROJECT_ROOT / config.specification_location)
    )

    auto_rest_test = AutoRestTest(
        spec_dir=specification_directory,
        config=config,
        tui=tui,
    )
    auto_rest_test.run_single(specification_name, ext)


if __name__ == "__main__":
    main()
