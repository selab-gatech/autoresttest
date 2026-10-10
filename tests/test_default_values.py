import random
import sys
import unittest
from pathlib import Path

from autoresttest.models import OperationProperties, SchemaProperties

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner, run_once  # noqa: E402

JSON = "application/json"
OPERATION = OperationProperties(
    operation_id="createUser",
    endpoint_path="/users",
    http_method="post",
    request_body={
        JSON: SchemaProperties(
            type="object",
            properties={
                "name": SchemaProperties(type="string"),
                "age": SchemaProperties(type="integer"),
                "meta": SchemaProperties(),  # no type
            },
            required=["name", "age"],
        )
    },
)


class DefaultBodyTests(unittest.TestCase):
    def test_object_body_has_every_property(self):
        learner = make_learner(OPERATION)
        # The seeds cover both typed branches for name and age; meta has no type.
        for seed in range(50):
            random.seed(seed)
            _, body = learner.generate_default_values("createUser")
            self.assertEqual(set(body[JSON]), {"name", "age", "meta"})

    def test_default_source_sends_every_selected_property(self):
        learner = make_learner(OPERATION)
        call = run_once(
            learner,
            "createUser",
            req_params=None,
            mime_type=JSON,
            data_source="DEFAULT",
            body_properties=("name", "age", "meta"),
        )
        body = call.args[2]
        self.assertEqual(set(body[JSON]), {"name", "age", "meta"})

    def test_default_source_sends_the_required_subset(self):
        # Before the fix the generated body held only "meta", so this subset sent {}.
        learner = make_learner(OPERATION)
        call = run_once(
            learner,
            "createUser",
            req_params=None,
            mime_type=JSON,
            data_source="DEFAULT",
            body_properties=("name", "age"),
        )
        self.assertEqual(set(call.args[2][JSON]), {"name", "age"})


if __name__ == "__main__":
    unittest.main()
