import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.models import ParameterKey
from autoresttest.utils import get_param_combinations, get_required_params


@dataclass
class ParameterAction:
    req_params: Optional[Tuple[ParameterKey, ...]]
    mime_type: Optional[str]


class ParameterAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        # Q-table maps the operation IDs to a dictionary containing a location key (e.g., "params") mapped to the parameter combinations
        self.q_table: dict[
            str, dict[str, dict[tuple[ParameterKey, ...] | str, float]]
        ] = {}
        self.operation_graph = operation_graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def initialize_q_table(self) -> None:
        for (
            operation_id,
            operation_node,
        ) in self.operation_graph.operation_nodes.items():
            if operation_id not in self.q_table:
                self.q_table[operation_id] = {"params": {}, "body": {}}
            required = get_required_params(
                operation_node.operation_properties.parameters
            )
            params = get_param_combinations(
                operation_node.operation_properties.parameters,
                required_params=required,
                seed=operation_id,
                config=self.operation_graph.config,
            )
            self.q_table[operation_id]["params"] = {param: 0.0 for param in params}
            mimes = (
                list(operation_node.operation_properties.request_body.keys())
                if operation_node.operation_properties.request_body
                else []
            )
            self.q_table[operation_id]["body"] = {mime: 0 for mime in mimes}
            self.q_table[operation_id]["params"]["None"] = 0
            self.q_table[operation_id]["body"]["None"] = 0

    def get_action(self, operation_id: str) -> ParameterAction:
        if operation_id not in self.q_table:
            raise ValueError(
                f"Operation '{operation_id}' not found in the Q-table for ParameterAgent."
            )
        if operation_id not in self.operation_graph.operation_nodes:
            raise ValueError(
                f"Operation '{operation_id}' not found in the operation graph."
            )
        if random.random() < self.epsilon:
            return self.get_random_action(operation_id)
        return self.get_best_action(operation_id)

    def get_random_action(self, operation_id: str) -> ParameterAction:
        random_params: tuple[ParameterKey, ...] | None = None
        if self.q_table[operation_id]["params"]:
            key = random.choice(list(self.q_table[operation_id]["params"].keys()))
            if key != "None" and isinstance(key, tuple):
                random_params = key

        random_mime: str | None = None
        if self.q_table[operation_id]["body"]:
            key = random.choice(list(self.q_table[operation_id]["body"].keys()))
            if key != "None" and isinstance(key, str):
                random_mime = key

        return ParameterAction(req_params=random_params, mime_type=random_mime)

    def get_best_action(self, operation_id: str) -> ParameterAction:
        required_params: set[ParameterKey] = get_required_params(
            self.operation_graph.operation_nodes[
                operation_id
            ].operation_properties.parameters
        )

        best_params: tuple[ParameterKey, ...] | None = None
        best_params_score: float = -np.inf

        if self.q_table[operation_id]["params"] and required_params:
            for params, score in self.q_table[operation_id]["params"].items():
                if (
                    params != "None"
                    and isinstance(params, tuple)
                    and required_params.issubset(set(params))
                    and score > best_params_score
                ):
                    best_params = params
                    best_params_score = score
        elif self.q_table[operation_id]["params"]:
            best_key = max(
                self.q_table[operation_id]["params"].items(), key=lambda x: x[1]
            )[0]
            if best_key != "None" and isinstance(best_key, tuple):
                best_params = best_key

        best_body: str | None = None
        best_body_score: float = -np.inf

        if self.q_table[operation_id]["body"]:
            for mime, score in self.q_table[operation_id]["body"].items():
                if mime != "None" and isinstance(mime, str) and score > best_body_score:
                    best_body = mime
                    best_body_score = score

        return ParameterAction(req_params=best_params, mime_type=best_body)

    def get_Q_next(self, operation_id: str) -> Tuple[float, float]:
        if operation_id not in self.q_table:
            return 0.0, 0.0
        best_next_q_params = (
            max(self.q_table[operation_id]["params"].values())
            if self.q_table[operation_id]["params"]
            else 0.0
        )
        best_next_q_body = (
            max(self.q_table[operation_id]["body"].values())
            if self.q_table[operation_id]["body"]
            else 0.0
        )
        return best_next_q_params, best_next_q_body

    def get_Q_curr(
        self, operation_id: str, action: ParameterAction
    ) -> Tuple[float, float]:
        if operation_id not in self.q_table:
            return 0.0, 0.0
        req_params_key = action.req_params if action.req_params is not None else "None"
        mime_type_key = action.mime_type if action.mime_type is not None else "None"
        current_q_params = (
            self.q_table[operation_id]["params"].get(req_params_key, 0.0)
            if self.q_table[operation_id]["params"]
            else 0.0
        )
        current_q_body = (
            self.q_table[operation_id]["body"].get(mime_type_key, 0.0)
            if self.q_table[operation_id]["body"]
            else 0.0
        )
        return current_q_params, current_q_body

    def update_Q_item(
        self, operation_id: str, action: ParameterAction, td_error: float
    ) -> None:
        if operation_id not in self.q_table:
            return
        req_params_key = action.req_params if action.req_params is not None else "None"
        mime_type_key = action.mime_type if action.mime_type is not None else "None"
        if (
            self.q_table[operation_id]["params"]
            and req_params_key in self.q_table[operation_id]["params"]
        ):
            self.q_table[operation_id]["params"][req_params_key] += (
                self.alpha * td_error
            )
        if (
            self.q_table[operation_id]["body"]
            and mime_type_key in self.q_table[operation_id]["body"]
        ):
            self.q_table[operation_id]["body"][mime_type_key] += self.alpha * td_error

    def update_q_table(
        self, operation_id: str, action: ParameterAction, reward: float
    ) -> None:
        if operation_id not in self.q_table:
            return
        req_params_key = action.req_params if action.req_params is not None else "None"
        mime_type_key = action.mime_type if action.mime_type is not None else "None"
        current_q_params = (
            self.q_table[operation_id]["params"].get(req_params_key, 0.0)
            if self.q_table[operation_id]["params"]
            else 0.0
        )
        current_q_body = (
            self.q_table[operation_id]["body"].get(mime_type_key, 0.0)
            if self.q_table[operation_id]["body"]
            else 0.0
        )
        best_next_q_params = (
            max(self.q_table[operation_id]["params"].values())
            if self.q_table[operation_id]["params"]
            else 0.0
        )
        best_next_q_body = (
            max(self.q_table[operation_id]["body"].values())
            if self.q_table[operation_id]["body"]
            else 0.0
        )
        new_q_params = current_q_params + self.alpha * (
            reward + self.gamma * best_next_q_params - current_q_params
        )
        new_q_body = current_q_body + self.alpha * (
            reward + self.gamma * best_next_q_body - current_q_body
        )
        if (
            self.q_table[operation_id]["params"]
            and req_params_key in self.q_table[operation_id]["params"]
        ):
            self.q_table[operation_id]["params"][req_params_key] = new_q_params
        if (
            self.q_table[operation_id]["body"]
            and mime_type_key in self.q_table[operation_id]["body"]
        ):
            self.q_table[operation_id]["body"][mime_type_key] = new_q_body

    def number_of_zeros(self, operation_id: str) -> int:
        if operation_id not in self.q_table:
            return 0
        zeros = 0
        for value in self.q_table[operation_id]["params"].values():
            if value == 0:
                zeros += 1
        for value in self.q_table[operation_id]["body"].values():
            if value == 0:
                zeros += 1
        return zeros
