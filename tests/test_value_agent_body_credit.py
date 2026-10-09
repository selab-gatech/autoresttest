import sys
import unittest
from pathlib import Path

from autoresttest.models import OperationProperties, SchemaProperties, ValueAction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner, run_once  # noqa: E402

JSON, XML = "application/json", "application/xml"
OPERATION = OperationProperties(
    operation_id="createItem",
    endpoint_path="/items",
    http_method="post",
    request_body={
        JSON: SchemaProperties(
            type="object", properties={"name": SchemaProperties(type="string")}
        ),
        XML: SchemaProperties(type="string"),
    },
)
LLM_VALUES = ValueAction(param_mappings=None, body_mappings={JSON: {"name": "x"}, XML: "<x/>"})


class ValueAgentBodyCreditTests(unittest.TestCase):
    def body_q_values(self, mime_type):
        learner = make_learner(OPERATION)
        learner.value_agent.q_table = {
            "createItem": {
                "params": {},
                "body": {JSON: [[{"name": "x"}, 0.0]], XML: [["<x/>", 0.0]]},
            }
        }
        run_once(
            learner,
            "createItem",
            req_params=None,
            mime_type=mime_type,
            data_source="LLM",
            value_action=LLM_VALUES,
            body_properties=("name",),
            status_code=400,
        )
        body = learner.value_agent.q_table["createItem"]["body"]
        return body[JSON][0][1], body[XML][0][1]

    def test_only_the_sent_body_is_credited(self):
        json_q, xml_q = self.body_q_values(JSON)
        self.assertNotEqual(json_q, 0.0)
        self.assertEqual(xml_q, 0.0)

    def test_no_body_is_credited_when_none_was_sent(self):
        self.assertEqual(self.body_q_values(None), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
