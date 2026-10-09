import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from autoresttest.graph import OperationGraph
from autoresttest.graph.similarity_comparator import OperationDependencyComparator
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    ResponseProperties,
    SchemaProperties,
)

# Unit vectors: "id" and "identifier" are close, "color" is unrelated.
VECTORS = {
    "id": np.array([1.0, 0.0]),
    "identifier": np.array([0.95, 0.31]),
    "color": np.array([0.0, 1.0]),
}


class StubModel:
    def handle_word_cases(self, word):
        return word.lower()

    def encode_sentence_or_word(self, text):
        return VECTORS.get(text)


def operation(operation_id, param, response_field):
    return OperationProperties(
        operation_id=operation_id,
        endpoint_path=f"/{operation_id}",
        http_method="get",
        parameters={
            (param, "query"): ParameterProperties(
                name=param, in_value="query", schema=SchemaProperties(type="string")
            )
        },
        responses={
            "200": ResponseProperties(
                content={
                    "application/json": SchemaProperties(
                        type="object",
                        properties={response_field: SchemaProperties(type="string")},
                    )
                }
            )
        },
    )


class SimilarityEdgeTests(unittest.TestCase):
    def setUp(self):
        self.comparator = OperationDependencyComparator(StubModel())

    def test_only_matches_above_the_threshold_are_similar(self):
        similar, tentative = self.comparator.compare_cosine(
            operation("a", "id", "color"), operation("b", "color", "color")
        )
        self.assertEqual(similar, {})
        self.assertEqual([param for param, _ in tentative], [("id", "query")] * 2)

        similar, _ = self.comparator.compare_cosine(
            operation("a", "id", "color"), operation("b", "identifier", "color")
        )
        self.assertEqual(list(similar), [("id", "query")])
        self.assertEqual(
            [sv.dependent_val for sv in similar[("id", "query")]],
            [("identifier", "query")],
        )

    def test_unmatched_operations_fall_back_to_tentative_edges(self):
        graph = OperationGraph(
            "unused", "probe", SimpleNamespace(config=Mock()), Mock()
        )
        graph.dependency_comparator = self.comparator
        operations = {
            "a": operation("a", "id", "identifier"),
            "b": operation("b", "color", "color"),
        }
        for properties in operations.values():
            graph.add_operation_node(properties)
        graph.determine_dependencies(operations)
        self.assertEqual(graph.operation_edges, [])
        edges = graph.operation_nodes["a"].outgoing_edges
        self.assertEqual([edge.destination.operation_id for edge in edges], ["b"])
        self.assertIn(("id", "query"), edges[0].similar_parameters)


if __name__ == "__main__":
    unittest.main()
