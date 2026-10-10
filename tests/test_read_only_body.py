import sys
import unittest
from pathlib import Path

from autoresttest.models import OperationProperties, SchemaProperties
from autoresttest.utils import get_required_body_params

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner  # noqa: E402

JSON = "application/json"
# Pet-clinic's Specialty: id is read-only but also listed as required.
SPECIALTY = SchemaProperties(
    type="object",
    properties={
        "id": SchemaProperties(type="integer", read_only=True),
        "name": SchemaProperties(type="string"),
    },
    required=["id", "name"],
)
OPERATION = OperationProperties(
    operation_id="addSpecialty",
    endpoint_path="/specialties",
    http_method="post",
    request_body={JSON: SPECIALTY},
)


class ReadOnlyBodyTests(unittest.TestCase):
    def test_read_only_properties_are_not_required(self):
        self.assertEqual(get_required_body_params(SPECIALTY), {"name"})

    def test_array_items_skip_read_only_properties(self):
        body = SchemaProperties(type="array", items=SPECIALTY)
        self.assertEqual(get_required_body_params(body), {"name"})

    def test_body_agent_can_choose_with_or_without_read_only_property(self):
        agent = make_learner(OPERATION).body_object_agent
        q_values = agent.q_table["addSpecialty"][JSON]
        self.assertIn(("name",), q_values)
        self.assertIn(("name", "id"), q_values)  # required properties come first
        self.assertNotIn(("id",), q_values)  # name is still required

        q_values[("name",)] = 1.0
        self.assertEqual(agent.get_best_action("addSpecialty", JSON), ("name",))
        q_values[("name", "id")] = 2.0
        self.assertEqual(agent.get_best_action("addSpecialty", JSON), ("name", "id"))

    def test_only_read_only_required_properties_leave_nothing_required(self):
        body = SchemaProperties(
            type="object",
            properties={"id": SchemaProperties(type="integer", read_only=True)},
            required=["id"],
        )
        self.assertEqual(get_required_body_params(body), set())


if __name__ == "__main__":
    unittest.main()
