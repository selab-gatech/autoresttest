import json
import sys
import time
import unittest
from pathlib import Path

from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    ResponseProperties,
    SchemaProperties,
)
from autoresttest.utils import UniqueValues

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marl_harness import make_learner, run_once  # noqa: E402

PARAM = ("q", "query")
OPERATION = OperationProperties(
    operation_id="listOwners",
    endpoint_path="/owners",
    http_method="get",
    parameters={
        PARAM: ParameterProperties(
            name="q", in_value="query", schema=SchemaProperties(type="string")
        )
    },
    responses={
        "200": ResponseProperties(
            status_code="200",
            content={
                "application/json": SchemaProperties(
                    type="array",
                    items=SchemaProperties(
                        type="object",
                        properties={
                            "id": SchemaProperties(type="integer"),
                            "active": SchemaProperties(type="boolean"),
                        },
                    ),
                )
            },
        )
    },
)


def typed(values):
    """Values with their types, so that True and 1 compare as different."""
    return [
        (type(value), typed(value) if isinstance(value, list) else value)
        for value in values
    ]


class UniqueValuesTests(unittest.TestCase):
    def test_skips_equal_values_and_keeps_order(self):
        values = UniqueValues(["a", {"x": 1, "y": 2}, "a", {"y": 2, "x": 1}, [1, 2]])
        self.assertEqual(values, ["a", {"x": 1, "y": 2}, [1, 2]])
        self.assertFalse(values.add([1, 2]))
        self.assertTrue(values.add([2, 1]))

    def test_keeps_booleans_apart_from_numbers(self):
        # A list's `in` treats True == 1 and False == 0, which dropped one of each pair.
        values = UniqueValues([1, True, 0, False, 1.0, None, "1", [True], [1]])
        self.assertEqual(
            typed(values),
            typed([1, True, 0, False, 1.0, None, "1", [True], [1]]),
        )

    def test_dedupes_dicts_with_parameter_key_tuples(self):
        signature = {"parameters": {("id", "path"): 5}, "body": None}
        values = UniqueValues(
            [signature, {"body": None, "parameters": {("id", "path"): 5}}]
        )
        self.assertEqual(len(values), 1)

    def test_is_still_a_json_serializable_list(self):
        values = UniqueValues([{"a": 1}, "b"])
        self.assertIsInstance(values, list)
        self.assertEqual(json.loads(json.dumps({"k": values})), {"k": [{"a": 1}, "b"]})


class ResponseRecordingTests(unittest.TestCase):
    def test_response_values_are_recorded_once(self):
        learner = make_learner(OPERATION)
        content = json.dumps(
            [
                {"id": 1, "active": True},
                {"id": 1, "active": 1},
                {"id": 2, "active": True},
            ]
        ).encode()
        run_once(
            learner,
            "listOwners",
            req_params=(PARAM,),
            data_source="DEFAULT",
            content=content,
        )
        recorded = learner.successful_responses["listOwners"]
        self.assertEqual(recorded["id"], [1, 2])
        self.assertEqual(typed(recorded["active"]), typed([True, 1]))

    def test_repeated_server_errors_are_one_unique_error(self):
        learner = make_learner(OPERATION)
        learner.generate_default_values = lambda operation_id: ({PARAM: "x"}, {})
        for _ in range(2):
            run_once(
                learner,
                "listOwners",
                req_params=(PARAM,),
                data_source="DEFAULT",
                status_code=500,
            )
        self.assertEqual(learner.errors["listOwners"], 2)
        self.assertEqual(len(learner.unique_errors["listOwners"]), 1)

    def test_large_collection_response_is_processed_in_linear_time(self):
        learner = make_learner(OPERATION)
        owners = [
            {"id": i, "name": f"owner{i}", "pets": [{"id": i, "type": {"id": 1}}]}
            for i in range(20000)
        ]
        mappings = {}
        start = time.perf_counter()
        learner._deconstruct_response(owners, mappings)
        # The list-scan version took 7-10 s on this input, the set-backed one about 0.1 s.
        self.assertLess(time.perf_counter() - start, 2)
        self.assertEqual(len(mappings["name"]), 20000)
        self.assertEqual(len(mappings["id"]), 20000)  # owner and pet ids overlap


if __name__ == "__main__":
    unittest.main()
