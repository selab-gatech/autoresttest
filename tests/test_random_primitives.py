import random
import sys
import unittest
from pathlib import Path

from autoresttest.models import OperationProperties

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner  # noqa: E402

OPERATION = OperationProperties(
    operation_id="getItem", endpoint_path="/items", http_method="get"
)


class RandomPrimitiveTests(unittest.TestCase):
    def test_single_returned_values_are_assigned(self):
        learner = make_learner(OPERATION)
        learner.successful_primitives = {
            "listTags": ["red", "green"],
            "countItems": [42],
            "getItem": ["own value"],
        }
        random.seed(1)
        for _ in range(50):
            parameters, body = learner.assign_random_from_primitives(
                {("tag", "query"): "x", ("id", "path"): 1},
                {"application/json": {"a": 1}},
                "getItem",
            )
            for value in [*parameters.values(), *body.values()]:
                self.assertIn(value, ["x", 1, {"a": 1}, "red", "green", 42])


if __name__ == "__main__":
    unittest.main()
