import random
from typing import TYPE_CHECKING, Any, Optional, TypedDict, cast

import numpy as np
from scipy.spatial.distance import cosine

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.models import ParameterKey, is_parameter_key
from autoresttest.utils import get_body_params


class DependentInfo(TypedDict):
    """Type for dependency information returned by get_best_action and get_random_action."""

    dependent_val: ParameterKey | str | None
    dependent_operation: str | None
    value: float
    in_value: str | None


if TYPE_CHECKING:
    from autoresttest.marl import QLearning

# Type alias for the inner Q-table structure: dependent_param -> Q-value
# dependent_param can be either str (body property or response) or ParameterKey (parameter)
DepParamDict = dict[ParameterKey | str, float]
# Type alias for location bucket: location -> {dependent_param -> Q-value}
LocBucketDict = dict[str, DepParamDict]
# Type alias for dependent operation: operation_id -> {location -> {dependent_param -> Q-value}}
DepOpDict = dict[str, LocBucketDict]
# Type alias for parameter level: ParameterKey|str -> {operation_id -> ...}
ParamLevelDict = dict[ParameterKey | str, DepOpDict]


class DependencyAgent(BaseAgent):
    """
    Agent that learns inter-operation parameter dependencies.

    Q-table structure: operation_id -> {"params": {...}, "body": {...}}
    - "params" dict uses ParameterKey tuples as keys (for operation parameters)
    - "body" dict uses str keys (for request body property names)
    ParameterKey | str is intentional to support both key types in the same structure.
    """

    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.q_table: dict[str, dict[str, ParamLevelDict]] = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.dependencies_discovered: int = 0

    def _notify(self, message: str) -> None:
        """Increment the dependencies discovered counter."""
        self.dependencies_discovered += 1

    @staticmethod
    def _bucket_location(location: str, is_source: bool = False) -> str:
        """
        Normalize various parameter locations into the q_table buckets.
        Source locations default to 'params' unless explicitly 'body'.
        Destination locations can be 'params', 'body', or 'response'.
        """
        if location == "body":
            return "body"
        if not is_source and location == "response":
            return "response"
        return "params"

    @staticmethod
    def _param_label(param_key: ParameterKey | str) -> str:
        return param_key[0] if isinstance(param_key, tuple) else param_key

    def initialize_q_table(self) -> None:
        # NOTE: Could flip dependent operation id and curr parameter name around to group by operation if need-be
        for (
            operation_id,
            operation_node,
        ) in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                params_dict: ParamLevelDict = {}
                body_dict: ParamLevelDict = {}
                self.q_table[operation_id] = {"params": params_dict, "body": body_dict}

            for edge in operation_node.outgoing_edges:
                for parameter, similarities in edge.similar_parameters.items():
                    # parameter can be either "str" (body property) or "ParameterKey" (parameter)
                    for similarity in similarities:
                        processed_in_val = similarity.in_value.split(" to ")
                        src_loc = processed_in_val[0] if processed_in_val else "params"
                        dest_loc = (
                            processed_in_val[1]
                            if len(processed_in_val) > 1
                            else "params"
                        )
                        dependent_parameter = similarity.dependent_val
                        # dependent_parameter can either be a "str" (body property or response) or "ParameterKey" (parameter)
                        destination = edge.destination.operation_id

                        # Note: Body should be nested

                        source_bucket = self._bucket_location(src_loc, is_source=True)
                        dest_bucket = self._bucket_location(dest_loc)

                        if (
                            source_bucket == "params"
                            and parameter not in self.q_table[operation_id]["params"]
                        ):
                            dep_op: DepOpDict = {}
                            self.q_table[operation_id]["params"][parameter] = dep_op
                        elif (
                            source_bucket == "body"
                            and parameter not in self.q_table[operation_id]["body"]
                        ):
                            dep_op_body: DepOpDict = {}
                            self.q_table[operation_id]["body"][parameter] = dep_op_body

                        if (
                            source_bucket == "params"
                            and destination
                            not in self.q_table[operation_id]["params"][parameter]
                        ):
                            loc_bucket: LocBucketDict = {
                                "params": {},
                                "body": {},
                                "response": {},
                            }
                            self.q_table[operation_id]["params"][parameter][
                                destination
                            ] = loc_bucket
                        elif (
                            source_bucket == "body"
                            and destination
                            not in self.q_table[operation_id]["body"][parameter]
                        ):
                            loc_bucket_body: LocBucketDict = {
                                "params": {},
                                "body": {},
                                "response": {},
                            }
                            self.q_table[operation_id]["body"][parameter][
                                destination
                            ] = loc_bucket_body

                        if source_bucket == "params":
                            self.q_table[operation_id]["params"][parameter][
                                destination
                            ][dest_bucket][dependent_parameter] = 0
                        elif source_bucket == "body":
                            self.q_table[operation_id]["body"][parameter][destination][
                                dest_bucket
                            ][dependent_parameter] = 0

    def get_action(
        self, operation_id: str, qlearning: "QLearning"
    ) -> tuple[str, dict[ParameterKey, Any], dict[str, Any]]:
        if operation_id not in self.q_table:
            raise ValueError(
                f"Operation '{operation_id}' not found in the Q-table for DependencyAgent."
            )
        has_success = any(
            status_code // 100 == 2
            for status_codes in qlearning.operation_response_counter.values()
            for status_code in status_codes
        )

        if random.random() < self.epsilon:
            if has_success and random.random() < 0.3:
                return self.assign_random_dependency_from_successful(
                    operation_id, qlearning
                )
            return self.get_random_action(operation_id, qlearning)

        return self.get_best_action(operation_id, qlearning)

    def get_best_action(
        self, operation_id: str, qlearning: "QLearning"
    ) -> tuple[str, dict[ParameterKey, Any], dict[str, Any]]:
        """ "Returns 'BEST', the parameter mapping, and body mapping."""
        successful_responses = qlearning.successful_responses
        successful_params = qlearning.successful_parameters
        successful_body = qlearning.successful_bodies

        best_params: dict[ParameterKey, DependentInfo] = {}
        for param, dependent_ops in self.q_table[operation_id]["params"].items():
            best_dependent: DependentInfo = {
                "dependent_val": None,
                "dependent_operation": None,
                "value": float(-np.inf),
                "in_value": None,
            }
            for dependent_op, value_dict in dependent_ops.items():
                for location, loc_params in value_dict.items():
                    for dependent_param, value in loc_params.items():
                        if (
                            value > best_dependent["value"]
                            and location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent["value"]
                            and location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent["value"]
                            and location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            best_dependent = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
            # Param for "params" should always be ParameterKey type
            best_params[cast(ParameterKey, param)] = best_dependent

        best_body: dict[str, DependentInfo] = {}
        for param, dependent_ops in self.q_table[operation_id]["body"].items():
            best_dependent_body: DependentInfo = {
                "dependent_val": None,
                "dependent_operation": None,
                "value": float(-np.inf),
                "in_value": None,
            }
            for dependent_op, value_dict in dependent_ops.items():
                for location, loc_params in value_dict.items():
                    for dependent_param, value in loc_params.items():
                        if (
                            value > best_dependent_body["value"]
                            and location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            best_dependent_body = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent_body["value"]
                            and location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            best_dependent_body = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
                        elif (
                            value > best_dependent_body["value"]
                            and location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            best_dependent_body = {
                                "dependent_val": dependent_param,
                                "dependent_operation": dependent_op,
                                "value": value,
                                "in_value": location,
                            }
            # param for body should always be string
            best_body[cast(str, param)] = best_dependent_body

        return "BEST", best_params, best_body

    def get_random_action(
        self, operation_id: str, qlearning: "QLearning"
    ) -> tuple[str, dict[ParameterKey, Any], dict[str, Any]]:
        """ "Returns 'EXPLORE', the parameter mapping, and body mapping."""
        successful_responses = qlearning.successful_responses
        successful_params = qlearning.successful_parameters
        successful_body = qlearning.successful_bodies

        random_params: dict[ParameterKey, DependentInfo] = {}
        for param, dependent_ops in self.q_table[operation_id]["params"].items():
            random_dependencies: list[DependentInfo] = []
            for dependent_op, value_dict in dependent_ops.items():
                for location, loc_params in value_dict.items():
                    for dependent_param, value in loc_params.items():
                        if (
                            location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            random_dependencies.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
            default_dep: DependentInfo = {
                "dependent_val": None,
                "dependent_operation": None,
                "value": 0.0,
                "in_value": None,
            }
            random_params[cast(ParameterKey, param)] = (
                random.choice(random_dependencies)
                if random_dependencies
                else default_dep
            )

        random_body: dict[str, DependentInfo] = {}
        for param, dependent_ops in self.q_table[operation_id]["body"].items():
            random_dependencies_body: list[DependentInfo] = []
            for dependent_op, value_dict in dependent_ops.items():
                for location, loc_params in value_dict.items():
                    for dependent_param, value in loc_params.items():
                        if (
                            location == "response"
                            and dependent_op in successful_responses
                            and dependent_param in successful_responses[dependent_op]
                            and successful_responses[dependent_op][dependent_param]
                        ):
                            random_dependencies_body.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "params"
                            and dependent_op in successful_params
                            and dependent_param in successful_params[dependent_op]
                            and successful_params[dependent_op][dependent_param]
                        ):
                            random_dependencies_body.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
                        elif (
                            location == "body"
                            and dependent_op in successful_body
                            and dependent_param in successful_body[dependent_op]
                            and successful_body[dependent_op][dependent_param]
                        ):
                            random_dependencies_body.append(
                                {
                                    "dependent_val": dependent_param,
                                    "dependent_operation": dependent_op,
                                    "value": value,
                                    "in_value": location,
                                }
                            )
            default_dep_body: DependentInfo = {
                "dependent_val": None,
                "dependent_operation": None,
                "value": 0.0,
                "in_value": None,
            }
            random_body[cast(str, param)] = (
                random.choice(random_dependencies_body)
                if random_dependencies_body
                else default_dep_body
            )

        return "EXPLORE", random_params, random_body

    def get_Q_next(
        self,
        operation_id: str,
        dependent_params: dict[ParameterKey | str, dict[str, Any]] | None,
        dependent_body: dict[str, dict[str, Any]] | None,
    ) -> tuple[list[float], list[float]]:
        best_next_q_params: list[float] = []
        best_next_q_body: list[float] = []

        if operation_id not in self.q_table:
            return best_next_q_params, best_next_q_body

        if dependent_params:
            for param, dependent in dependent_params.items():
                best_next_q: float = -np.inf
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("params", {}):
                    continue
                if (
                    dependent["dependent_operation"]
                    not in self.q_table[operation_id]["params"][param]
                ):
                    continue
                dep_op_dict = self.q_table[operation_id]["params"][param][
                    dependent["dependent_operation"]
                ]
                for location, loc_params in dep_op_dict.items():
                    for dependent_param, value in loc_params.items():
                        best_next_q = max(best_next_q, value)
                best_next_q_params.append(best_next_q)

        if dependent_body:
            for param, dependent in dependent_body.items():
                best_next_q = float(-np.inf)
                if not dependent["dependent_operation"]:
                    continue
                if param not in self.q_table[operation_id].get("body", {}):
                    continue
                if (
                    dependent["dependent_operation"]
                    not in self.q_table[operation_id]["body"][param]
                ):
                    continue
                dep_op_dict = self.q_table[operation_id]["body"][param][
                    dependent["dependent_operation"]
                ]
                for location, loc_params in dep_op_dict.items():
                    for dependent_param, value in loc_params.items():
                        best_next_q = max(best_next_q, value)
                best_next_q_body.append(best_next_q)

        return best_next_q_params, best_next_q_body

    def _dependency_bucket(
        self, operation_id: str, section: str, param: ParameterKey | str, dependent
    ) -> DepParamDict | None:
        """
        Return the location bucket that holds the Q-value of `dependent`, or None.
        The bucket is chosen by the dependency's in_value, so a body property and a
        response field with the same name keep separate Q-values.
        """
        if operation_id not in self.q_table or not dependent["dependent_operation"]:
            return None
        dep_op_dict = (
            self.q_table[operation_id]
            .get(section, {})
            .get(param, {})
            .get(dependent["dependent_operation"])
        )
        if not dep_op_dict:
            return None
        bucket = dep_op_dict.get(dependent["in_value"])
        if bucket is None or dependent["dependent_val"] not in bucket:
            return None
        return bucket

    def get_Q_curr(
        self,
        operation_id: str,
        dependent_params: dict[ParameterKey | str, dict[str, Any]] | None,
        dependent_body: dict[str, dict[str, Any]] | None,
    ) -> tuple[list[float], list[float]]:
        current_Q_params: list[float] = []
        current_Q_body: list[float] = []
        for section, dependents, current_Q in (
            ("params", dependent_params, current_Q_params),
            ("body", dependent_body, current_Q_body),
        ):
            for param, dependent in (dependents or {}).items():
                bucket = self._dependency_bucket(operation_id, section, param, dependent)
                if bucket is not None:
                    current_Q.append(bucket[dependent["dependent_val"]])
        return current_Q_params, current_Q_body

    def update_Q_item(
        self,
        operation_id: str,
        dependent_params: dict[ParameterKey | str, dict[str, Any]] | None,
        dependent_body: dict[str, dict[str, Any]] | None,
        td_error: float,
    ) -> None:
        for section, dependents in (
            ("params", dependent_params),
            ("body", dependent_body),
        ):
            for param, dependent in (dependents or {}).items():
                bucket = self._dependency_bucket(operation_id, section, param, dependent)
                if bucket is not None:
                    bucket[dependent["dependent_val"]] += self.alpha * td_error

    def add_undocumented_responses(
        self, new_operation_response_id: str, new_property: str
    ) -> bool:
        updated_tables = False
        dependency_comparator = self.operation_graph.dependency_comparator
        embedding_model = self.operation_graph.embedding_model
        for operation_id, operation_props in self.q_table.items():
            for location, param_values in operation_props.items():
                for param, dependent_values in param_values.items():
                    processed_param = embedding_model.handle_word_cases(
                        self._param_label(param)
                    )
                    processed_response = embedding_model.handle_word_cases(new_property)
                    param_embedding = embedding_model.encode_sentence_or_word(
                        processed_param
                    )
                    response_embedding = embedding_model.encode_sentence_or_word(
                        processed_response
                    )
                    if param_embedding is not None and response_embedding is not None:
                        similarity = 1 - cosine(param_embedding, response_embedding)
                        if similarity > dependency_comparator.threshold:
                            if new_operation_response_id not in dependent_values:
                                dependent_values[new_operation_response_id] = {}
                            if (
                                "response"
                                not in dependent_values[new_operation_response_id]
                            ):
                                dependent_values[new_operation_response_id][
                                    "response"
                                ] = {}
                            dependent_values[new_operation_response_id]["response"][
                                new_property
                            ] = 0
                            updated_tables = True
                            self._notify(
                                f"Dependency: {operation_id} → {new_operation_response_id} ({param} → {new_property})"
                            )
        return updated_tables

    def add_new_dependency(
        self,
        operation_id: str,
        param_location: str,
        operation_param: ParameterKey | str,
        dependent_operation_id: str,
        dependent_location: str,
        dependent_param: str,
    ) -> None:
        # Validate type matches location to maintain Q-table invariants
        if param_location == "params" and not is_parameter_key(operation_param):
            print(
                f"Warning: Expected ParameterKey for 'params', got {type(operation_param).__name__}. Skipping dependency."
            )
            return
        if param_location == "body" and not isinstance(operation_param, str):
            print(
                f"Warning: Expected str for 'body', got {type(operation_param).__name__}. Skipping dependency."
            )
            return

        if operation_param not in self.q_table[operation_id][param_location]:
            self.q_table[operation_id][param_location][operation_param] = {}
        if (
            dependent_operation_id
            not in self.q_table[operation_id][param_location][operation_param]
        ):
            self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ] = {"params": {}, "body": {}, "response": {}}

        # Ensure dependent_location is valid
        if dependent_location not in ["params", "body", "response"]:
            print(
                f"Warning: Invalid dependent_location '{dependent_location}'. Skipping dependency."
            )
            return

        # Ensure the dependent_location key exists in the structure
        if (
            dependent_location
            not in self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ]
        ):
            self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ][dependent_location] = {}

        if (
            dependent_param
            not in self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ][dependent_location]
        ):
            self.q_table[operation_id][param_location][operation_param][
                dependent_operation_id
            ][dependent_location][dependent_param] = 0
            self._notify(
                f"Dependency: {operation_id} → {dependent_operation_id} ({operation_param} → {dependent_param})"
            )

    # Get a random value from the successful operations to test dependencies
    def assign_random_dependency_from_successful(
        self, operation_id: str, qlearning: "QLearning"
    ) -> tuple[str, dict[ParameterKey, Any], dict[str, Any]]:
        """Returns 'RANDOM', the parameter mapping, and body mapping"""
        possible_options = []

        for (
            operation_idx,
            operation_parameters,
        ) in qlearning.successful_parameters.items():
            if operation_idx == operation_id:
                continue
            for parameter_key, parameter_values in operation_parameters.items():
                for parameter_value in parameter_values:
                    possible_options.append(
                        {
                            "dependent_val": parameter_key,
                            "dependent_operation": operation_idx,
                            "value": parameter_value,
                            "in_value": "params",
                        }
                    )

        for operation_idx, operation_body_parms in qlearning.successful_bodies.items():
            if operation_idx == operation_id:
                continue
            for body_name, body_values in operation_body_parms.items():
                for body_value in body_values:
                    possible_options.append(
                        {
                            "dependent_val": body_name,
                            "dependent_operation": operation_idx,
                            "value": body_value,
                            "in_value": "body",
                        }
                    )

        for (
            operation_idx,
            operation_responses,
        ) in qlearning.successful_responses.items():
            if operation_idx == operation_id:
                continue
            for response_name, response_values in operation_responses.items():
                for response_value in response_values:
                    possible_options.append(
                        {
                            "dependent_val": response_name,
                            "dependent_operation": operation_idx,
                            "value": response_value,
                            "in_value": "response",
                        }
                    )

        if not possible_options:
            return "RANDOM", {}, {}

        parameter_dependency_assignment = {}
        op_props = qlearning.operation_graph.operation_nodes[
            operation_id
        ].operation_properties
        if op_props.parameters:
            for (
                parameter_name,
                parameter_properties,
            ) in op_props.parameters.items():
                if parameter_properties.schema:
                    parameter_dependency_assignment[parameter_name] = random.choice(
                        possible_options
                    )

        body_dependency_assignment = {}
        if op_props.request_body:
            for mime, body_properties in op_props.request_body.items():
                possible_body_params = get_body_params(body_properties)
                for prop in possible_body_params:
                    body_dependency_assignment[prop] = random.choice(possible_options)

        return "RANDOM", parameter_dependency_assignment, body_dependency_assignment

    def number_of_zeros(self, operation_id: str) -> int:
        if operation_id not in self.q_table:
            return 0
        zeros = 0
        for location, param_values in self.q_table[operation_id].items():
            for param, dependent_values in param_values.items():
                for dependent_op, dependent_props in dependent_values.items():
                    for dependent_location, dependent_params in dependent_props.items():
                        for dependent_param, value in dependent_params.items():
                            if value == 0:
                                zeros += 1
        return zeros
