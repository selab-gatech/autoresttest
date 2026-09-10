# AutoRestTest

### Introduction
***AutoRestTest*** is a complete testing software for automated API testing that combines the utility of graph
theory, Large Language Models (LLMs), and multi-agent reinforcement learning (MARL) to parse the OpenAPI Specification
and create enhanced comprehensive test cases. AutoRestTest specifically supports **OpenAPI Specification 3.0**.

Watch this [demonstration video](https://www.youtube.com/watch?v=VVus2W8rap8) of AutoRestTest to learn how it solves complex challenges in automated REST API testing, as well as its configuration, execution steps, and output.

> [!NOTE]
> Following the release of the demonstration video, the code base has been refactored.
> Refer to this `README.md` for the most current setup and execution details.

<p align="center">
  <a href="https://www.youtube.com/watch?v=VVus2W8rap8">
    <img src="https://img.youtube.com/vi/VVus2W8rap8/0.jpg" alt="Watch the video">
  </a>
</p>

The program uses LLMs for natural-language processing during the creation of the reinforcement
learning tables and graph edges. AutoRestTest supports any LLM from an OpenAI-API compatible provider, including OpenAI, OpenRouter, Azure OpenAI, and local models (LocalAI, LM Studio, vLLM, Ollama, etc.).

> [!Important]
> An API key from your chosen LLM provider is required. The cost per execution depends on your provider and model choice.
> For reference, when testing an average API with ~15 operations using GPT-4o-mini, the cost was approximately $0.1.

## Terminal User Interface (TUI)

AutoRestTest features a modern, interactive terminal user interface built with [Rich](https://github.com/Textualize/rich) that provides:

### Interactive Configuration Wizard

On startup, AutoRestTest launches an interactive configuration wizard that allows you to:
- **Select API specifications** from discovered files or enter custom paths
- **Choose LLM providers** (OpenAI, OpenRouter, Local) with pre-configured model options
- **Configure test duration** with convenient presets (5, 10, 20, 30, 60 minutes)
- **Adjust Q-learning parameters** (learning rate, discount factor, exploration)
- **Toggle caching options** for faster repeated runs
- **Override API URLs** for local testing

The wizard uses sensible defaults from `configurations.toml`, so you can simply press Enter to accept defaults or customize any setting.

### Live Execution Dashboard

During request generation, a real-time dashboard displays:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ [STATUS] Request Generation                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │ Time Elapsed: 00:15:32            Time Remaining: 00:04:28           │    ║
║  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.8%    │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  Successfully Processed (2xx) Operations:          1,482                     ║
║  Operation Coverage:                               85.2%                     ║
║  Unique Server Errors (5xx):                       7                         ║
║  Total Requests Sent:                              3,847                     ║
║                                                                               ║
║  Status Code Distribution                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │ 200: ████████████████████████████████████████████░░░░░░░░  1,482    │    ║
║  │ 404: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    150    │    ║
║  │ 500: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      7    │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  Current Operation: GET /api/users/{id}                                       ║
║                                                                               ║
║  [COST] Total LLM usage: $0.11 USD                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

Features include:
- **Real-time progress tracking** with elapsed and remaining time
- **Color-coded status codes** (green for 2xx, orange for 4xx, red for 5xx)
- **Visual progress bars** for status code distribution
- **Live cost estimation** based on token usage
- **Current operation indicator** showing what's being tested
- **Mutation counter** tracking the number of fuzzing mutations applied

### Q-Table Initialization Progress

During the Q-table initialization phase, the Value Agent and Header Agent (if enabled) perform LLM calls to generate test values for each API operation. A live progress display shows:

```
╭──── ⚙ Value Agent Q-Table Generation ────╮
│                                           │
│   ━━━━━━━━━━━━━━━━━━━━━━━───────  67.5%  │
│                                           │
│   Operations: 27/40  │  Elapsed: 03:45   │
│                                           │
│   ▶ POST /api/users                       │
│                                           │
╰───────────────────────────────────────────╯
```

This progress display appears for both Value Agent and Header Agent initialization, showing:
- Visual progress bar with percentage complete
- Operation count (completed/total)
- Elapsed time
- Current operation being processed

### TUI Command Line Options

| Option | Description |
|--------|-------------|
| `--skip-wizard` | Skip configuration wizard, use `configurations.toml` directly |
| `--quick` | Quick setup wizard (essential settings only) |
| `-s, --spec PATH` | Override specification path |
| `-t, --time SECONDS` | Override test duration |
| `--width N` | Set TUI display width (default: 100) |

**Examples:**

```bash
# Full interactive mode (default)
poetry run autoresttest

# Quick setup - only essential settings
poetry run autoresttest --quick

# Skip wizard entirely, use config file
poetry run autoresttest --skip-wizard

# Override spec and duration via CLI
poetry run autoresttest -s specs/original/oas/spotify.yaml -t 600
```

## Installation

We recommend using Poetry with `pyproject.toml` for dependency management and scripts. A `poetry.lock` file pins exact versions.

Steps:
1. Clone the repository.
2. Ensure Python 3.10.x is available (project targets `>=3.10,<3.11`).
3. Install dependencies with Poetry (uses `poetry.lock` if present):
   - `poetry install`
4. Copy `configurations.toml.example` to `configurations.toml` and customize the specification and provider settings:
   - `cp configurations.toml.example configurations.toml`
5. Create a `.env` file in the project root and add:
   - `API_KEY='<YOUR_API_KEY>'`

Alternatives (provided but not recommended):
- `pip install -r requirements.txt`
- `conda env create -f autoresttest.yaml`

Optionally, if the user wants to test specific APIs, they can install their OpenAPI Specification files within a folder
in the root directory. For convenience, we have provided a large array of OpenAPI Specifications for popular and 
widely-used APIs. These Specifications can be seen in the `aratrl-openapi` and `specs` directories.

### Running Local Services

AutoRestTest includes support for running local REST API services for testing. Some services require building before use:

#### JDK 8_1 Services (features-service, ncs, scs)

These services must be built before first use:

```bash
# Make the build script executable
chmod +x services/build_jdk8_1_services.sh

# Build all JDK 8_1 services
bash services/build_jdk8_1_services.sh
```

After building, start a service:
```bash
cd services
python3 run_service_mac.py features-service no_token
```

For detailed instructions, see `FEATURES_SERVICE_SETUP.md`.

#### Other Supported Services

Other services like genome-nexus, language-tool, youtube, etc. can be run without building:
```bash
cd services
python3 run_service_mac.py <service-name> <token>
```

See `services/README.md` for the full list of supported services.

At this point, the installation step is complete and the software can be executed. However, it is important that the
following configuration steps are completed for purposeful execution.

## Configuration

There is a wide array of configuration options available within the codebase. All configuration options are easily accessible via a single TOML file at the project root: `configurations.toml`.

Start with the tracked [configurations.toml.example](configurations.toml.example):

```bash
cp configurations.toml.example configurations.toml
```

Edit the local copy for your API and LLM provider. `configurations.toml` is ignored by Git; keep reusable defaults in the example and credentials in `.env`. The example is a starting point and is not loaded automatically.

If you already have a `configurations.toml`, keep your settings rather than replacing it with the example. Back it up before updating an older checkout where the file was tracked, then restore it after updating. The new timeout settings have defaults, so existing configuration files continue to work without edits.

### Configuration precedence

Each run starts by loading `configurations.toml`. Overrides are applied in this order, from lowest to highest priority:

| Priority | Source | Behavior |
|----------|--------|----------|
| 1 | `configurations.toml` | Supplies the base settings. |
| 2 | Configuration wizard | Replaces only settings changed in the full or quick wizard. Optional with `--skip-wizard`. |
| 3 | Explicit CLI options | `--spec` and `--time` replace the corresponding settings after the wizard. |

Overrides merge individual settings within each section. For example, changing the model in the wizard and passing `--time 600` keeps the chosen model and changes only the testing duration. Settings omitted from the wizard or CLI retain their earlier values, including explicit `false` and `0` values where those values are valid. The testing duration must be positive.

```bash
# File defaults, with optional wizard changes
poetry run autoresttest

# File defaults plus CLI overrides; skip the wizard
poetry run autoresttest --skip-wizard --spec specs/original/oas/spotify.yaml --time 600

# Wizard choices, with the CLI duration taking final priority
poetry run autoresttest --time 600
```

The configuration summary shows the resolved settings used by the run. The parser, agents, LLM clients, JSON repair, and API requests all receive that configuration, including the timeout settings. Wizard and CLI overrides stay in memory and do not modify `configurations.toml`. Edit the file to persist changes for future runs.

Below are the relevant settings and where to find them in `configurations.toml`.

#### 1. Specifying the API Specification

If you intend to use your own OpenAPI Specification file as described in the Installation section, set the relative path (from the project root) to that file in `configurations.toml` under `[spec].location`.

Only `.yaml` and `.json` files are supported (OpenAPI 3.0). Example: `aratrl-openapi/market2.yaml`.

Specification parsing is handled by [Prance](https://github.com/jfinkhaeuser/prance). If your spec has circular/self-referencing `$ref` chains, you can tune the resolver behavior with:
- `[spec].recursion_limit` (default: `50`) — the maximum number of times a circular reference may appear in the resolution stack before a placeholder schema is substituted; a value of 1 means a self-referencing element is resolved once before being replaced.
- `[spec].strict_validation` (default: `true`) — when `true`, the OpenAPI spec is strictly validated and parsing stops on errors; when `false`, invalid sections are skipped where possible so execution can continue.

#### 2. Configuring Reinforcement Learning Parameters

Configure Q-learning parameters in `configurations.toml`:
- `[q_learning].learning_rate` (default: `0.1`)
- `[q_learning].discount_factor` (default: `0.9`)
- `[q_learning].max_exploration` (epsilon; default: `1`, decays over time to `0.1`)

Instead of limiting episodes, the program limits RL iterations using a time budget. Set the duration (in seconds) under:
- `[request_generation].time_duration` (default: `1200`)

> [!NOTE]
> This time budget applies only to the MARL (Q-learning) phase. The initial value table generation phase, which runs before Q-learning begins, is not time-limited and will process all operations in the specification.

Requests and retry waits consume the existing testing budget; they never increase `time_duration`. The runner uses a monotonic deadline and starts no further request attempts after it expires. An in-flight request can finish after the deadline: connection/read timeouts bound network inactivity, not the total time to receive a response. Initial generation uses the same network timeouts but has no overall phase deadline.

#### Parameter Combination Sampling

For APIs with many parameters, generating all possible combinations would cause memory exhaustion. AutoRestTest uses depth-weighted stratified sampling to bound combinations while prioritizing smaller (more diagnostic) combinations. Configure under `[agent]`:

| Option | Default | Description |
|--------|---------|-------------|
| `max_combinations` | `12` | Maximum optional parameters per combination. Required parameters are always included. |
| `max_total_combinations` | `3000` | Hard cap on combinations per operation. Smaller combinations are kept when truncating. |
| `base_samples_per_size` | `200` | Sample count at size=1; decays for larger sizes (e.g., size=5 gets ~60 samples). |
| `combination_seed` | `42` | Seed for reproducible random sampling. Change for different samples. |

Example distribution for an operation with 30 parameters:
- Size 1-2: All combinations enumerated (30 + 435 = 465)
- Size 3-12: Random samples (~700 total)
- Total: ~1,200 unique combinations (well under 3K cap)

> [!TIP]
> The above steps alter the variables across all agents used within the software. If the user desires to change
> the individual agent parameters, they can navigate to the `src/autoresttest/agents` files and change the parameters.

#### 3. Parallel Value Table Generation

The Value Agent's Q-table initialization can be parallelized using a thread pool, which significantly speeds up startup for APIs with many operations. Configure this under `[agent.value]`:
- `[agent.value].parallelize` (default: `true`) — when `true`, operations are processed concurrently using a thread pool; when `false`, the original sequential depth-first traversal is used.
- `[agent.value].max_workers` (default: `4`) — number of worker threads for parallel generation (ignored if `parallelize` is `false`). Set to match your CPU core count for optimal performance.

> [!NOTE]
> Rate-limited responses (HTTP 429) are automatically retried with exponential backoff.

#### 4. Configuring the LLM

AutoRestTest supports any LLM from an OpenAI-API compatible provider. Configure the model and provider in `configurations.toml`:

| Option | Default | Description |
|--------|---------|-------------|
| `[llm].engine` | `gpt-4o-mini` | Model identifier. Format depends on provider (e.g., `gpt-4o-mini`, `google/gemini-2.0-flash-001`, `anthropic/claude-3-haiku`). |
| `[llm].api_base` | `https://api.openai.com/v1` | API endpoint URL. Change for alternative providers. |
| `[llm].creative_temperature` | `1` | Temperature for creative parameter generation. |
| `[llm].strict_temperature` | `1` | Temperature for repair or deterministic flows. |
| `[llm].max_tokens` | `20000` | Maximum tokens for LLM response. Set to `-1` to omit and use provider default. |
| `[llm].timeout_seconds` | `120.0` | Positive, finite network timeout for model-provider calls. |

The LLM SDK retries eligible failures up to twice. AutoRestTest does not wrap those attempts in an additional retry loop. The timeout applies to network operations within each attempt, so a call including retries can take longer than `timeout_seconds`.

**Example configurations:**

```toml
# OpenAI (default)
[llm]
engine = "gpt-4o-mini"
api_base = "https://api.openai.com/v1"

# OpenRouter (access to many models)
[llm]
engine = "google/gemini-2.0-flash-001"
api_base = "https://openrouter.ai/api/v1"

# Local model (LM Studio, Ollama, etc.)
[llm]
engine = "local-model"
api_base = "http://localhost:1234/v1"
```

> [!WARNING]
> The model must support **JSON mode**. Most recent models from major providers support this feature.
> The console output will list token usage for analyzing costs, if supplied by the provider.

#### 5. Use of Cached Graphs and Reinforcement Learning Tables

The software can cache the graph and Q-tables to reduce cost and speed up repeated runs. Configure this behavior under:
- `[cache].use_cached_graph` (example: `false`)
- `[cache].use_cached_table` (example: `false`)

> [!IMPORTANT]
> The cache parameter structure changed on Dec. 10, 2025. Recreate caches (clear `cache/` contents) so they remain compatible with newer runs.

> [!NOTE]
> When enabled, these options store and read cached data under the `cache/` directory at the project root (for example, `cache/graphs/` and `cache/q_tables/`).
> If disk usage becomes a concern, you can delete the cached project information in `cache/` after use; the data will be regenerated on the next run when needed.

#### 6. Optional Header Agent

AutoRestTest contains an optional Header agent responsible for testing the API with different authentication headers. Due to difficulties 
of different authentication flows, the Header agent is only able to use Basic Authentication. By default, the agent is disabled.

Toggle via `configurations.toml`:
- `[agents.header].enabled` (default: `false`)

> [!CAUTION]
> The Header agent Q-table should be rerun when executing services with local databases that refresh, as the user
> credentials may become invalid.

#### 7. Custom Static Headers

For APIs that require custom headers (such as API keys or Bearer tokens), you can configure static headers in the `[custom_headers]` section. These headers are included with every request.

```toml
[custom_headers]
X-API-Key = "your-static-key"
Authorization = "Bearer ${ACCESS_TOKEN}"
```

Environment variable interpolation is supported using the `${VAR_NAME}` syntax. Store sensitive values in your `.env` file:
```
ACCESS_TOKEN=your_bearer_token
```

> [!CAUTION]
> If you enable the Header Agent (`[agents.header].enabled = true`) alongside custom headers, be aware that the Header Agent may override your `Authorization` header during testing. The Header Agent uses Basic Authentication tokens and has higher priority than custom static headers. For Bearer token or API key authentication, keep the Header Agent disabled.

#### 8. Custom API URL Override

By default, AutoRestTest extracts the API base URL from the `servers` field in your OpenAPI specification. If you need to target a different host (e.g., a local development server or staging environment), you can override this behavior in the `[api]` section:

```toml
[api]
override_url = true
host = "localhost"
port = 8080
```

When `override_url` is set to `true`, the tool constructs the API URL as `http://{host}:{port}/` instead of using the specification's server URL.

| Option | Default | Description |
|--------|---------|-------------|
| `override_url` | `false` | When `false`, uses the URL from the OpenAPI spec. When `true`, uses the custom host and port. |
| `host` | `localhost` | The hostname for the custom API URL. |
| `port` | `8080` | The port number for the custom API URL. |
| `request_timeout_seconds` | `30.0` | Positive, finite connection/read inactivity timeout for requests to the API being tested, including setup requests. Applies regardless of `override_url`. |

Configure the two network paths independently:

```toml
[llm]
timeout_seconds = 120.0

[api]
request_timeout_seconds = 30.0
```

Add these options to the existing sections in your local configuration. During testing, request timeouts and 429 retry waits are limited by the remaining phase budget. During setup, a 429 response whose retry delay exceeds `request_timeout_seconds` is returned without another attempt, avoiding a long wait or an early retry against the server's instructions.

> [!TIP]
> This is useful when testing local services that run on a different port than specified in the OpenAPI spec, or when the spec contains a production URL but you want to test against a local or staging environment.

## Execution

Run the script using Poetry, after following the installation instructions:
```
poetry run autoresttest
```

To indicate the specification for execution, set `[spec].location` in `configurations.toml`. This path must be relative to the project root.

### Docker Execution

For ease of use, the software can be executed using Docker. The user can apply the following commands from the 
root directory sequentially to execute AutoRestTest using the pre-built Docker image:

```
docker build -t autoresttest .
docker run -it autoresttest
```

Ensure that `configurations.toml` is configured correctly before executing the software.

## Results

Throughout AutoRestTest's execution, the TUI provides real-time visual feedback including:
- **Phase indicators** showing current execution stage (Graph Construction, Q-Table Initialization, Request Generation)
- **Live dashboard** during request generation with time tracking, status code distribution, and cost estimation
- **Final report** with summary statistics, success rates, and token usage
- **Smart status code grouping** - automatically groups by category (2xx, 4xx, 5xx) when there are many different codes
- **Color-coded time indicators** - remaining time changes color based on percentage remaining (red < 5%, yellow < 20%) 

#### Report Generation and Data Files

AutoRestTest will generate a series of files in the `data` directory within the root folder for easy access to execution results. 
The following are the files generated:
- `report.json`: Contains the compiled report of the API testing process, including key summary statistics and information.
- `operation_status_codes.json`: Contains the distribution of status codes for each operation.
- `q_tables.json`: Contains the completed Q-tables for each agent.
- `server_errors.json`: Contains the unique server errors encountered during the testing process. Only JSON-seriable errors are stored.
- `successful_parameters.json`: Contains the successful parameters for each operation.
- `successful_responses.json`: Contains the successful responses for each operation.
- `successful_bodies.json`: Contains the successful object request bodies for each operation.
- `successful_primitives.json`: Contains the successful primitive request bodies for each operation.

These files contain the necessary information for analysis into the success of AutoRestTest. 

> [!NOTE]
> The output files can grow in size according to the number of operations and the duration of execution. 
> For example, the file containing successful responses can grow to be several **gigabytes** when executing for a long duration. 
> It is recommended to clear the `data` directory when the files are no longer needed.
