import random

import numpy as np

from .base_agent import BaseAgent

from autoresttest.graph import OperationGraph
from autoresttest.utils import get_combinations, get_required_body_params


class BodyObjAgent(BaseAgent):
    def __init__(
        self,
        operation_graph: OperationGraph,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        # Q-table maps the operation ID to a dictionary that maps the MIME type to a dictionary where the key is the body property combinations, or "None", and the value is a float
        self.q_table: dict[str, dict[str, dict[tuple[str, ...] | str, float]]] = {}
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
                self.q_table[operation_id] = {}
            if operation_node.operation_properties.request_body:
                for (
                    mime,
                    body_properties,
                ) in operation_node.operation_properties.request_body.items():
                    if body_properties.type == "object" and body_properties.properties:
                        required_body = get_required_body_params(body_properties)
                        body_obj_combinations = get_combinations(
                            body_properties.properties.keys(),
                            required=required_body,
                            seed=f"{operation_id}:{mime}",
                            config=self.operation_graph.config,
                        )
                        self.q_table[operation_id][mime] = {
                            body_obj: 0.0 for body_obj in body_obj_combinations
                        }
                        self.q_table[operation_id][mime]["None"] = 0.0

    def get_action(self, operation_id: str, mime: str) -> tuple[str, ...] | None:
        if operation_id not in self.q_table:
            raise ValueError(
                f"Operation '{operation_id}' not found in the Q-table for BodyObjAgent."
            )
        if mime not in self.q_table.get(operation_id, {}):
            raise ValueError(
                f"MIME type '{mime}' not found for operation '{operation_id}' in the Q-table for BodyObjAgent."
            )
        if random.random() < self.epsilon:
            return self.get_random_action(operation_id, mime)
        return self.get_best_action(operation_id, mime)

    def get_best_action(self, operation_id: str, mime: str) -> tuple[str, ...] | None:
        request_body = self.operation_graph.operation_nodes[
            operation_id
        ].operation_properties.request_body
        if request_body is None:
            return None
        required_obj_params = get_required_body_params(request_body[mime])
        result: tuple[str, ...] | None = None
        if required_obj_params:
            best_value = -np.inf
            for body_obj, value in self.q_table[operation_id][mime].items():
                if (
                    isinstance(body_obj, tuple)
                    and value > best_value
                    and required_obj_params.issubset(set(body_obj))
                ):
                    result = body_obj
                    best_value = value
        else:
            if self.q_table[operation_id][mime]:
                best_key = max(
                    self.q_table[operation_id][mime].items(), key=lambda x: x[1]
                )[0]
                if isinstance(best_key, tuple):
                    result = best_key
        return result

    def get_random_action(self, operation_id: str, mime: str) -> tuple[str, ...] | None:
        if not self.q_table[operation_id][mime]:
            return None
        action = random.choice(list(self.q_table[operation_id][mime].keys()))
        if isinstance(action, tuple):
            return action
        return None

    def update_q_table(
        self,
        operation_id: str,
        mime: str,
        action: tuple[str, ...] | None,
        reward: float,
    ) -> None:
        key: tuple[str, ...] | str = action if action is not None else "None"
        current_q = (
            self.q_table[operation_id][mime][key]
            if self.q_table[operation_id][mime]
            else 0.0
        )
        best_next_q = (
            max(self.q_table[operation_id][mime].values())
            if self.q_table[operation_id][mime]
            else 0.0
        )
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[operation_id][mime][key] = new_q

    def get_Q_next(self, operation_id: str, mime: str) -> float:
        if operation_id not in self.q_table or mime not in self.q_table.get(
            operation_id, {}
        ):
            return 0.0
        return (
            max(self.q_table[operation_id][mime].values())
            if self.q_table[operation_id][mime]
            else 0.0
        )

    def get_Q_curr(
        self, operation_id: str, mime: str, action: tuple[str, ...] | None
    ) -> float:
        key: tuple[str, ...] | str = action if action is not None else "None"
        if operation_id not in self.q_table or mime not in self.q_table.get(
            operation_id, {}
        ):
            return 0.0
        return (
            self.q_table[operation_id][mime].get(key, 0.0)
            if self.q_table[operation_id][mime]
            else 0.0
        )

    def update_Q_item(
        self,
        operation_id: str,
        mime: str,
        action: tuple[str, ...] | None,
        td_error: float,
    ) -> None:
        key: tuple[str, ...] | str = action if action is not None else "None"
        if operation_id not in self.q_table or mime not in self.q_table.get(
            operation_id, {}
        ):
            return
        if key not in self.q_table[operation_id][mime]:
            return
        self.q_table[operation_id][mime][key] += self.alpha * td_error

    def number_of_zeros(self, operation_id: str) -> int:
        zeros = 0
        for body_obj_mappings in self.q_table.get(operation_id, {}).values():
            for value in body_obj_mappings.values():
                if value == 0:
                    zeros += 1
        return zeros
