import unittest
from types import SimpleNamespace

from autoresttest.agents import ValueAgent
from autoresttest.models import ValueAction


class ValueAgentQNextTests(unittest.TestCase):
    def test_each_parameter_and_body_uses_its_own_best_value(self):
        agent = ValueAgent(SimpleNamespace(operation_nodes={}))
        agent.q_table = {
            "op": {
                "params": {
                    ("a", "query"): [["x", 5.0], ["y", 0.0]],
                    ("b", "query"): [["x", -1.0], ["y", -2.0]],
                    ("c", "query"): [["x", -1.5]],
                    ("d", "query"): [],
                },
                "body": {
                    "application/json": [[{"k": 1}, 3.0]],
                    "application/xml": [["<k/>", -4.0]],
                },
            }
        }
        action = ValueAction(
            param_mappings={
                ("a", "query"): "x",
                ("b", "query"): "x",
                ("c", "query"): "x",
                ("d", "query"): "x",
                ("unknown", "query"): "x",
            },
            body_mappings={"application/json": {"k": 1}, "application/xml": "<k/>"},
        )
        self.assertEqual(
            agent.get_Q_next("op", action), ([5.0, -1.0, -1.5, 0.0], [3.0, -4.0])
        )


if __name__ == "__main__":
    unittest.main()
