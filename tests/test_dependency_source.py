import sys
import unittest
from pathlib import Path

from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
    ValueAction,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner, run_once  # noqa: E402

PARAM = ("id", "path")
OPERATION = OperationProperties(
    operation_id="getItem",
    endpoint_path="/items/{id}",
    http_method="get",
    parameters={
        PARAM: ParameterProperties(
            name="id",
            in_value="path",
            required=True,
            schema=SchemaProperties(type="integer"),
        )
    },
)
LLM_VALUES = ValueAction(param_mappings={PARAM: 5}, body_mappings=None)


class DependencySourceParameterTests(unittest.TestCase):
    def test_parameters_are_filled_without_parameter_dependencies(self):
        call = run_once(
            make_learner(OPERATION),
            "getItem",
            req_params=(PARAM,),
            data_source="DEPENDENCY",
            value_action=LLM_VALUES,
        )
        self.assertEqual(call.args[1], {PARAM: 5})

    def test_falsy_dependency_values_are_kept(self):
        learner = make_learner(OPERATION)
        learner.successful_responses["listItems"] = {"id": [0]}
        dependency = {
            PARAM: {
                "dependent_operation": "listItems",
                "dependent_val": "id",
                "in_value": "response",
            }
        }
        call = run_once(
            learner,
            "getItem",
            req_params=(PARAM,),
            data_source="DEPENDENCY",
            dependency_action=("BEST", dependency, {}),
            value_action=LLM_VALUES,
        )
        self.assertEqual(call.args[1], {PARAM: 0})


if __name__ == "__main__":
    unittest.main()
