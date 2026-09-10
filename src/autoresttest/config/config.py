"""Configuration loading utilities for AutoRestTest."""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import tomli as tomllib
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Load environment variables from .env file for custom header interpolation
load_dotenv()


CONFIG_FILE_NAME = "configurations.toml"
CONFIG_DIR = Path(__file__).resolve().parent
AUTORESTTEST_DIR = CONFIG_DIR.parent
PROJECT_ROOT = AUTORESTTEST_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / CONFIG_FILE_NAME


class SpecConfig(BaseModel):
    location: str
    recursion_limit: int = 50
    strict_validation: bool = True


class LLMConfig(BaseModel):
    engine: str
    creative_temperature: float
    strict_temperature: float
    api_base: str = "https://api.openai.com/v1"
    max_tokens: int = 20000
    timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)


class HeaderAgentConfig(BaseModel):
    enabled: bool


class AgentsConfig(BaseModel):
    header: HeaderAgentConfig


class ValueAgentConfig(BaseModel):
    parallelize: bool = True
    max_workers: int = 4


class AgentCombinationConfig(BaseModel):
    max_combinations: int = 12
    max_total_combinations: int = 3000
    base_samples_per_size: int = 200
    combination_seed: int = 42
    value: ValueAgentConfig = ValueAgentConfig()


class CacheConfig(BaseModel):
    use_cached_graph: bool
    use_cached_table: bool


class QLearningConfig(BaseModel):
    learning_rate: float
    discount_factor: float
    max_exploration: float


class RequestGenerationConfig(BaseModel):
    time_duration: int = Field(gt=0)
    mutation_rate: float


class ApiConfig(BaseModel):
    """API URL configuration. Override the spec URL with custom host/port."""

    override_url: bool = False
    host: str = "localhost"
    port: int = 8080
    request_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)


class CustomHeadersConfig(BaseModel):
    """Custom static headers. Supports ${VAR_NAME} env var interpolation."""

    model_config = ConfigDict(extra="allow")

    def get_headers(self) -> Dict[str, str]:
        headers = {}
        for key, value in (self.model_extra or {}).items():
            if isinstance(value, str):
                headers[key] = re.sub(
                    r"\$\{([^}]+)\}", lambda m: os.getenv(m.group(1), ""), value
                )
            else:
                headers[key] = str(value)
        return headers


class Config(BaseModel):
    spec: SpecConfig
    llm: LLMConfig
    agents: AgentsConfig
    agent: AgentCombinationConfig
    cache: CacheConfig
    q_learning: QLearningConfig
    request_generation: RequestGenerationConfig
    api: ApiConfig = ApiConfig()
    custom_headers: CustomHeadersConfig = CustomHeadersConfig()

    model_config = ConfigDict(frozen=True)

    @property
    def specification_location(self) -> str:
        return self.spec.location

    @property
    def recursion_limit(self) -> int:
        return self.spec.recursion_limit

    @property
    def strict_validation(self) -> bool:
        return self.spec.strict_validation

    @property
    def openai_llm_engine(self) -> str:
        return self.llm.engine

    @property
    def creative_temperature(self) -> float:
        return self.llm.creative_temperature

    @property
    def strict_temperature(self) -> float:
        return self.llm.strict_temperature

    @property
    def llm_api_base(self) -> str:
        """Return LLM API base URL."""
        return self.llm.api_base

    @property
    def llm_max_tokens(self) -> int:
        """Return LLM max tokens. -1 means omit from API call."""
        return self.llm.max_tokens

    @property
    def enable_header_agent(self) -> bool:
        return self.agents.header.enabled

    @property
    def max_combinations(self) -> int:
        return self.agent.max_combinations

    @property
    def max_total_combinations(self) -> int:
        return self.agent.max_total_combinations

    @property
    def base_samples_per_size(self) -> int:
        return self.agent.base_samples_per_size

    @property
    def combination_seed(self) -> int:
        return self.agent.combination_seed

    @property
    def parallelize_value_generation(self) -> bool:
        return self.agent.value.parallelize

    @property
    def value_generation_workers(self) -> int:
        return self.agent.value.max_workers

    @property
    def static_headers(self) -> Dict[str, str]:
        """Return custom headers with env var interpolation applied."""
        return self.custom_headers.get_headers()

    @property
    def custom_api_url(self) -> str:
        """Construct API URL from host and port."""
        return f"http://{self.api.host}:{self.api.port}/"


def _load_raw_config(path: Path | str | None = None) -> Dict[str, Any]:
    config_path = Path(path) if path is not None else CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Copy configurations.toml.example to configurations.toml and edit it before running."
        )
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: Path | str | None = None) -> Config:
    """Load file defaults for a new run, without reusing a previous run's values."""
    return Config.model_validate(_load_raw_config(path))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return cached file defaults for standalone callers without an explicit config."""
    return load_config()


def apply_config_overrides(
    overrides: Dict[str, Any], config: Config | None = None
) -> Config:
    """Merge only supplied fields into a config, validate, and leave inputs unchanged."""

    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    base = config if config is not None else get_config()
    return Config.model_validate(deep_merge(base.model_dump(), overrides))


__all__ = [
    "Config",
    "CONFIG_PATH",
    "PROJECT_ROOT",
    "get_config",
    "load_config",
    "apply_config_overrides",
]
