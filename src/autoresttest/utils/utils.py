import base64
import hashlib
import itertools
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import numpy as np
from dotenv import load_dotenv
from gensim.downloader import load
from gensim.models import KeyedVectors

from autoresttest.config import Config, get_config
from autoresttest.models import ParameterKey, ParameterProperties, SchemaProperties
from autoresttest.prompts.generator_prompts import FIX_JSON_OBJ
from autoresttest.prompts.system_prompts import FIX_JSON_SYSTEM_MESSAGE
from autoresttest.specification import SpecificationParser

load_dotenv()

CACHE_ROOT = Path(__file__).resolve().parent.parents[2] / "cache"
Q_TABLE_CACHE_DIR = CACHE_ROOT / "q_tables"
GRAPH_CACHE_DIR = CACHE_ROOT / "graphs"


def remove_nulls(item: Any) -> Any:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    elif isinstance(item, dict):
        cleaned = {k: remove_nulls(v) for k, v in item.items() if v}
        return {k: v for k, v in cleaned.items() if v}
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        cleaned = [remove_nulls(i) for i in item]
        return [i for i in cleaned if i is not None]
    else:
        return item


def make_param_key(name: str | None, in_value: str | None) -> ParameterKey:
    """
    Build a canonical parameter key from name and in_value.
    """
    return (name or "", in_value or None)


def param_key_to_label(key: ParameterKey) -> str:
    """
    Create a stable string label for a parameter key (for JSON/LLM prompts).
    """
    name, in_value = key
    loc = in_value if in_value is not None else "unspecified"
    return f"{name}::{loc}"


def label_to_param_key(label: str) -> ParameterKey:
    """
    Convert a parameter label back into a key tuple.
    """
    if "::" in label:
        name, loc = label.split("::", 1)
        loc = None if loc == "unspecified" else loc
    else:
        name, loc = label, None
    return make_param_key(name, loc)


def get_param_combinations(
    operation_parameters: Dict[ParameterKey, ParameterProperties],
    required_params: Optional[Set[ParameterKey]] = None,
    seed: Optional[str] = None,
    config: Config | None = None,
) -> List[Tuple[ParameterKey, ...]]:
    param_list = get_params(operation_parameters)
    return get_combinations(
        param_list, required=required_params, seed=seed, config=config
    )


def get_body_combinations(
    operation_body: Dict[str, SchemaProperties],
    config: Config | None = None,
) -> Dict[str, List[Tuple[str]]]:
    return {
        k: get_combinations(v, config=config)
        for k, v in get_request_body_params(operation_body).items()
    }


def get_body_object_combinations(
    body_schema: SchemaProperties,
    required_body_params: Optional[Set[str]] = None,
    seed: Optional[str] = None,
    config: Config | None = None,
) -> List[Tuple[str, ...]]:
    return get_combinations(
        get_body_params(body_schema),
        required=required_body_params,
        seed=seed,
        config=config,
    )


def get_combinations(
    arr: Iterable[Any],
    required: Optional[Set[Any]] = None,
    seed: Optional[str] = None,
    config: Config | None = None,
) -> List[Tuple[Any, ...]]:
    """
    Generate bounded parameter combinations with depth-weighted sampling.

    Uses stratified sampling that prioritizes smaller combinations while ensuring
    required parameters are always included. For large parameter sets, random
    sampling is used with seeded RNG for reproducibility.

    Args:
        arr: All parameters to combine.
        required: Parameters that must appear in every combination.
        seed: Seed string for reproducible randomness (e.g., operation ID).

    Returns:
        List of parameter combination tuples.
    """
    arr = list(arr) if arr is not None else []
    required = required or set()
    optional = [p for p in arr if p not in required]
    required_tuple = tuple(p for p in arr if p in required)  # Preserve order

    config = config if config is not None else get_config()
    max_optional_size = config.max_combinations
    max_total = config.max_total_combinations
    base_samples = config.base_samples_per_size

    # Seeded RNG for reproducibility
    if seed:
        seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed_int)
    else:
        rng = random.Random(config.combination_seed)

    combinations: Set[Tuple[Any, ...]] = set()
    n_optional = len(optional)

    # Always include: required-only and all-params
    combinations.add(required_tuple)
    if optional:
        combinations.add(required_tuple + tuple(optional))

    if n_optional <= max_optional_size:
        # Small enough: exhaustive enumeration of optional params
        for size in range(1, n_optional + 1):
            for combo in itertools.combinations(optional, size):
                combinations.add(required_tuple + combo)
    else:
        # Large: depth-weighted sampling (smaller sizes get more samples)
        for size in range(1, min(max_optional_size, n_optional) + 1):
            # Exponential decay: size=1 gets base_samples, larger sizes get fewer
            samples_for_size = max(10, int(base_samples / (size**0.7)))
            total_possible = math.comb(n_optional, size)

            if total_possible <= samples_for_size:
                # Small enough to enumerate all
                for combo in itertools.combinations(optional, size):
                    combinations.add(required_tuple + combo)
            else:
                # Random sample with seeded RNG
                sampled: Set[Tuple[Any, ...]] = set()
                attempts = 0
                max_attempts = samples_for_size * 20
                while len(sampled) < samples_for_size and attempts < max_attempts:
                    indices = rng.sample(range(n_optional), size)
                    combo = tuple(optional[i] for i in sorted(indices))
                    sampled.add(combo)
                    attempts += 1
                for combo in sampled:
                    combinations.add(required_tuple + combo)

    # Enforce hard cap (deterministic order: sort by size, then content)
    result = sorted(combinations, key=lambda x: (len(x), x))
    if len(result) > max_total:
        # Keep smallest combinations (most valuable for issue isolation)
        result = result[:max_total]

    return result


def get_params(
    operation_parameters: Dict[ParameterKey, ParameterProperties],
) -> List[ParameterKey]:
    return list(operation_parameters.keys()) if operation_parameters is not None else []


def get_required_params(
    operation_parameters: Dict[ParameterKey, ParameterProperties],
) -> Set[ParameterKey]:
    required_parameters = set()
    for parameter, parameter_properties in operation_parameters.items():
        if parameter_properties.required:
            required_parameters.add(parameter)
    return required_parameters


def get_required_body_params(operation_body: SchemaProperties) -> Optional[Set]:
    if operation_body is None:
        return None
    required_body = set()

    if operation_body.properties and operation_body.type == "object":
        for key, value in operation_body.properties.items():
            # Check if key is in the PARENT's required list (not child's required field)
            if operation_body.required and key in operation_body.required:
                required_body.add(key)

    elif operation_body.items and operation_body.type == "array":
        required_body = get_required_body_params(operation_body.items)

    else:
        return None
    return required_body


def encode_dict_as_key(dictionary: Dict) -> str:
    json_str = json.dumps(dictionary, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_body_params(body: SchemaProperties) -> List[str]:
    if body is None:
        return []

    elif body.properties and body.type == "object":
        body_params = []
        for key, value in body.properties.items():
            body_params.append(key)
        return body_params

    elif body.items and body.type == "array":
        return get_body_params(body.items)

    return []


def get_response_params(response: SchemaProperties, response_params: list[str]) -> None:
    if response is None:
        return

    if response.properties:
        for key, value in response.properties.items():
            if key not in response_params:
                response_params.append(key)
            get_response_params(value, response_params)

    elif response.items:
        get_response_params(response.items, response_params)


def get_response_param_mappings(
    response: SchemaProperties, response_mappings: dict[str, SchemaProperties]
) -> None:
    if response is None:
        return

    if response.properties:
        for key, value in response.properties.items():
            response_mappings[key] = value
            get_response_param_mappings(value, response_mappings)

    elif response.items:
        get_response_param_mappings(response.items, response_mappings)


def get_request_body_params(
    operation_body: Dict[str, SchemaProperties],
) -> Dict[str, List[str]]:
    return (
        {k: get_body_params(v) for k, v in operation_body.items()}
        if operation_body is not None
        else {}
    )


def split_parameter_values(
    operation_parameters: Dict[ParameterKey, ParameterProperties],
    provided_values: Optional[Dict[ParameterKey, Any]],
):
    """
    Split provided parameter values into path, query, header, and cookie buckets based on their 'in' value.
    Ignores parameters that are not defined on the operation.
    """
    path_params: Dict[str, Any] = {}
    query_params: Dict[str, Any] = {}
    header_params: Dict[str, Any] = {}
    cookie_params: Dict[str, Any] = {}

    if not provided_values:
        return path_params, query_params, header_params, cookie_params

    for key, value in provided_values.items():
        normalized_key = key
        if normalized_key not in operation_parameters and not isinstance(
            normalized_key, tuple
        ):
            # Fallback: match by name when provided without location
            for candidate_key in operation_parameters.keys():
                if (
                    isinstance(candidate_key, tuple)
                    and candidate_key[0] == normalized_key
                ):
                    normalized_key = candidate_key
                    break

        if normalized_key not in operation_parameters:
            continue
        if value is None:
            continue
        name, in_value = normalized_key
        in_value = in_value or operation_parameters[normalized_key].in_value

        if in_value == "path":
            path_params[name] = value
        elif in_value == "header":
            header_params[name] = value
        elif in_value == "cookie":
            cookie_params[name] = value
        else:
            query_params[name] = value

    return path_params, query_params, header_params, cookie_params


def get_object_shallow_mappings(thing: Any) -> Optional[Dict[str, Any]]:
    """
    Determine the mappings of a given item that contains some nested objects
    :param thing: The thing to get the mappings for
    :return:
    """
    if not thing:
        return None
    mappings = {}
    if type(thing) == dict:
        for key, value in thing.items():
            mappings[key] = value
    elif type(thing) == list and len(thing) > 0:
        mappings = get_object_shallow_mappings(thing[0])
    return mappings


def compose_json_fix_prompt(invalid_json_str: str):
    prompt = FIX_JSON_OBJ
    prompt += invalid_json_str
    return prompt


def attempt_fix_json(invalid_json_str: str, config: Config | None = None):
    from autoresttest.llm import LanguageModel

    config = config if config is not None else get_config()
    language_model = LanguageModel(temperature=config.strict_temperature, config=config)
    json_prompt = compose_json_fix_prompt(invalid_json_str)
    fixed_json = language_model.query(
        user_message=json_prompt, system_message=FIX_JSON_SYSTEM_MESSAGE, json_mode=True
    )
    try:
        fixed_json = json.loads(fixed_json)
        return fixed_json
    except (ValueError, RecursionError):
        print("Attempt to fix JSON string failed.")
        print(f"Original JSON string: {invalid_json_str}")
        print(f"Fixed JSON string: {fixed_json}")
        return {}


def _is_json_mime(mime_type: str) -> bool:
    """
    Returns True for any JSON-like MIME type.
    """
    if not mime_type:
        return False
    mime_lower = mime_type.lower()
    return (
        "json" in mime_lower
        or mime_lower.endswith("+json")
        or mime_lower.endswith("/json")
    )


def get_accept_header(responses: dict | None) -> str | None:
    """Extract Accept header from operation responses.

    Returns comma-separated MIME types from 2xx responses, or None.
    """
    if not responses:
        return None
    mime_types = set()
    for status_code, response_props in responses.items():
        if status_code and status_code.startswith("2") and response_props.content:
            mime_types.update(response_props.content.keys())
    return ", ".join(sorted(mime_types)) if mime_types else None


def _dispatch_request_inner(
    select_method,
    full_url: str,
    params: Dict,
    body: Dict[str, Any] | None,
    headers: Dict,
    cookies: Optional[Dict],
):
    """
    Internal helper that performs a single HTTP request.
    """
    if not body:
        return select_method(
            full_url, params=params, headers=headers or None, cookies=cookies
        )

    if not isinstance(body, dict):
        return select_method(
            full_url, params=params, data=body, headers=headers or None, cookies=cookies
        )

    # Use the first provided MIME type; bodies are expected to be singular.
    mime_type, payload = next(iter(body.items()))
    mime_lower = mime_type.lower() if mime_type else ""

    if _is_json_mime(mime_type):
        headers.setdefault("Content-Type", mime_type)
        if payload is not None:
            return select_method(
                full_url,
                params=params,
                json=payload,
                headers=headers or None,
                cookies=cookies,
            )
        return select_method(
            full_url, params=params, headers=headers or None, cookies=cookies
        )

    if "x-www-form-urlencoded" in mime_lower:
        headers.setdefault("Content-Type", mime_type)
        body_data = get_object_shallow_mappings(payload)
        if not body_data or not isinstance(body_data, dict):
            body_data = {"data": payload}
        return select_method(
            full_url,
            params=params,
            data=body_data,
            headers=headers or None,
            cookies=cookies,
        )

    if mime_lower.startswith("multipart/"):
        # Convert payload to proper files format for requests.
        # Each field must be a tuple: (filename, data) or (filename, data, content_type)
        # Using None as filename indicates a form field (not a file upload).
        files_data = {}
        if isinstance(payload, dict):
            for field_name, field_value in payload.items():
                if field_value is None:
                    continue
                # Serialize non-string/bytes values to JSON
                if isinstance(field_value, (str, bytes)):
                    serialized = field_value
                else:
                    serialized = json.dumps(field_value)
                files_data[field_name] = (None, serialized)
        else:
            # Non-dict payload: serialize entire thing
            files_data = {"data": (None, json.dumps(payload) if payload else "")}

        return select_method(
            full_url,
            params=params,
            files=files_data,
            headers=headers or None,
            cookies=cookies,
        )

    if mime_lower.startswith("text/"):
        headers.setdefault("Content-Type", mime_type)
        if not isinstance(payload, str):
            payload = str(payload)
        return select_method(
            full_url,
            params=params,
            data=payload,
            headers=headers or None,
            cookies=cookies,
        )

    # Fallback: send whatever the MIME type is with a best-effort serializer.
    headers.setdefault("Content-Type", mime_type)
    if isinstance(payload, (dict, list)):
        return select_method(
            full_url,
            params=params,
            json=payload,
            headers=headers or None,
            cookies=cookies,
        )
    return select_method(
        full_url, params=params, data=payload, headers=headers or None, cookies=cookies
    )


def dispatch_request(
    select_method,
    full_url: str,
    params: Dict,
    body: Dict[str, Any] | None,
    header: Optional[Dict] = None,
    cookies: Optional[Dict] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    accept: str | None = None,
    timeout_seconds: float | None = None,
    deadline: float | None = None,
    config: Config | None = None,
):
    """
    Send a request with sensible handling for the provided body and MIME type key (if any).
    Includes automatic retry with exponential backoff for rate-limited (429) responses.
    deadline, when supplied, is an absolute time.monotonic() testing deadline.
    """
    params = params or {}
    headers = header.copy() if header is not None else {}
    cookies = cookies or None
    if accept:
        headers.setdefault("Accept", accept)
    if timeout_seconds is None:
        config = config if config is not None else get_config()
        timeout_seconds = config.api.request_timeout_seconds

    response = None
    for attempt in range(max_retries + 1):
        request_timeout = timeout_seconds
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return response
            request_timeout = min(request_timeout, remaining)
        response = _dispatch_request_inner(
            partial(select_method, timeout=request_timeout),
            full_url,
            params,
            body,
            headers.copy(),
            cookies,
        )

        if response is None:
            return None

        # Handle rate limiting (429) with exponential backoff + jitter
        if response.status_code == 429:
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s + random jitter (0-1s)
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, int(retry_after))
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return response
                    delay = min(delay, remaining)
                elif delay > timeout_seconds:
                    # During setup, return the 429 rather than wait indefinitely
                    # or retry earlier than the server requested.
                    return response
                print(
                    f"Rate limited (429). Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue

        return response

    return response  # Return last response even if still 429


def encode_dictionary(dictionary) -> str:
    json_str = json.dumps(dictionary, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def is_json_seriable(data):
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False


class EmbeddingModel:
    def __init__(self):
        self.model: KeyedVectors = cast(KeyedVectors, load("glove-wiki-gigaword-50"))
        self.threshold = 0.8
        self._embedding_cache: Dict[str, Optional[np.ndarray]] = {}

    def encode_sentence_or_word(self, thing: str) -> Optional[np.ndarray]:
        if thing in self._embedding_cache:
            return self._embedding_cache[thing]

        words = thing.split(" ")
        word_vectors: list[np.ndarray] = [
            self.model[word] for word in words if word in self.model
        ]
        result = np.mean(word_vectors, axis=0) if word_vectors else None
        self._embedding_cache[thing] = result
        return result

    def clear_cache(self):
        """Clear embedding cache to free memory after graph generation."""
        self._embedding_cache.clear()

    @staticmethod
    def handle_word_cases(parameter):
        reconstructed_parameter = []
        for index, char in enumerate(parameter):
            if char == "_" or char == "-":
                reconstructed_parameter.append(" ")
            elif char.isalpha():
                if char.isupper() and index != 0:
                    reconstructed_parameter.append(" " + char.lower())
                else:
                    reconstructed_parameter.append(char)
        return "".join(reconstructed_parameter)


def construct_db_dir():
    for path in (Q_TABLE_CACHE_DIR, GRAPH_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_q_table_cache_path(spec_name: str) -> Path:
    construct_db_dir()
    return Q_TABLE_CACHE_DIR / spec_name


def get_graph_cache_path(spec_name: str) -> Path:
    construct_db_dir()
    return GRAPH_CACHE_DIR / spec_name


def construct_basic_token(token):
    username = token.get("username")
    password = token.get("password")
    token_str = f"{username}:{password}"
    encoded_bytes = base64.b64encode(token_str.encode("utf-8"))
    encoded_str = encoded_bytes.decode("utf-8")
    return f"Basic {encoded_str}"


def get_api_url(spec_parser: SpecificationParser):
    return spec_parser.get_api_url()
