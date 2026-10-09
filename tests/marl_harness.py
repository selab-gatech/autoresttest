"""Run one iteration of QLearning.execute_operations with stubbed agent choices."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from autoresttest.agents.parameter_agent import ParameterAction
from autoresttest.config import apply_config_overrides, load_config
from autoresttest.marl.marl import QLearning


def make_learner(operation):
    graph = SimpleNamespace(
        config=apply_config_overrides(
            {"agents": {"header": {"enabled": False}}},
            load_config(
                Path(__file__).resolve().parents[1] / "configurations.toml.example"
            ),
        ),
        request_generator=SimpleNamespace(api_url="http://example.invalid"),
        operation_nodes={
            operation.operation_id: SimpleNamespace(
                operation_properties=operation, outgoing_edges=[]
            )
        },
        operation_edges=[],
    )
    learner = QLearning(graph, time_duration=1, mutation_rate=0)
    for agent in (
        learner.operation_agent,
        learner.parameter_agent,
        learner.body_object_agent,
        learner.data_source_agent,
        learner.dependency_agent,
    ):
        agent.initialize_q_table()
    learner.data_source_agent.initialize_dependency_source()
    learner.value_agent.q_table = {
        operation.operation_id: {"params": {}, "body": {}}
    }
    return learner


def run_once(
    learner,
    operation_id,
    *,
    req_params,
    mime_type=None,
    data_source,
    dependency_action=("BEST", {}, {}),
    value_action=None,
    body_properties=None,
    status_code=200,
    content=b"{}",
):
    """Run one request and return the mocked send_operation call."""
    response = requests.Response()
    response.status_code, response._content = status_code, content
    patches = [
        patch("autoresttest.marl.marl.time.monotonic", side_effect=[0, 0, 2]),
        patch.object(learner.operation_agent, "get_action", return_value=operation_id),
        patch.object(
            learner.parameter_agent,
            "get_action",
            return_value=ParameterAction(req_params, mime_type),
        ),
        patch.object(learner.data_source_agent, "get_action", return_value=data_source),
        patch.object(
            learner.dependency_agent, "get_action", return_value=dependency_action
        ),
        patch.object(learner, "send_operation", return_value=response),
    ]
    if value_action is not None:
        patches += [
            patch.object(learner.value_agent, "get_action", return_value=value_action),
            patch.object(
                learner.value_agent, "get_best_action", return_value=value_action
            ),
        ]
    if body_properties is not None:
        patches.append(
            patch.object(
                learner.body_object_agent, "get_action", return_value=body_properties
            )
        )
    for active in patches:
        active.start()
    try:
        learner.execute_operations()
        return learner.send_operation.call_args
    finally:
        for active in reversed(patches):
            active.stop()
