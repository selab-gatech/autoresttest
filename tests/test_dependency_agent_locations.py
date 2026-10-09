import unittest
from types import SimpleNamespace

from autoresttest.agents import DependencyAgent

PARAM = ("userId", "path")


def make_agent():
    agent = DependencyAgent(SimpleNamespace(operation_nodes={}), alpha=0.5)
    # getUser's userId may come from createUser's request body "id" or its response "id".
    agent.q_table = {
        "getUser": {
            "params": {
                PARAM: {
                    "createUser": {
                        "params": {},
                        "body": {"id": 0.0},
                        "response": {"id": 0.0},
                    }
                }
            },
            "body": {},
        }
    }
    return agent


def dependency(in_value):
    return {
        PARAM: {
            "dependent_operation": "createUser",
            "dependent_val": "id",
            "in_value": in_value,
        }
    }


class DependencyLocationTests(unittest.TestCase):
    def test_updates_only_touch_the_dependency_location(self):
        agent = make_agent()
        agent.update_Q_item("getUser", dependency("response"), {}, td_error=5)
        buckets = agent.q_table["getUser"]["params"][PARAM]["createUser"]
        self.assertEqual(buckets["response"]["id"], 2.5)
        self.assertEqual(buckets["body"]["id"], 0.0)

    def test_current_q_reads_the_dependency_location(self):
        agent = make_agent()
        agent.update_Q_item("getUser", dependency("body"), {}, td_error=-2)
        self.assertEqual(agent.get_Q_curr("getUser", dependency("body"), {}), ([-1.0], []))
        self.assertEqual(
            agent.get_Q_curr("getUser", dependency("response"), {}), ([0.0], [])
        )

    def test_best_action_follows_the_rewarded_location(self):
        agent = make_agent()
        agent.update_Q_item("getUser", dependency("response"), {}, td_error=5)
        learner = SimpleNamespace(
            successful_responses={"createUser": {"id": [7]}},
            successful_parameters={},
            successful_bodies={"createUser": {"id": [1]}},
        )
        _, params, _ = agent.get_best_action("getUser", learner)
        self.assertEqual(params[PARAM]["in_value"], "response")

    def test_missing_entries_are_ignored(self):
        agent = make_agent()
        missing = {PARAM: {**dependency("params")[PARAM]}}
        agent.update_Q_item("getUser", missing, {}, td_error=5)
        agent.update_Q_item("unknownOp", dependency("body"), {}, td_error=5)
        self.assertEqual(agent.get_Q_curr("getUser", missing, {}), ([], []))
        buckets = agent.q_table["getUser"]["params"][PARAM]["createUser"]
        self.assertEqual(buckets["body"]["id"], 0.0)
        self.assertEqual(buckets["response"]["id"], 0.0)


if __name__ == "__main__":
    unittest.main()
