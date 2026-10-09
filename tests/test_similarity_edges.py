import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from autoresttest.agents import DataSourceAgent
from autoresttest.graph import OperationGraph
from autoresttest.graph.similarity_comparator import OperationDependencyComparator
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    SchemaProperties,
)

# Cosine similarities: id~identifier 0.95 (above the 0.8 threshold),
# id~key 0.70 (between the 0.5 floor and the threshold), id~color 0 and key~color 0.
VECTORS = {
    "id": np.array([1.0, 0.0, 0.0]),
    "identifier": np.array([0.95, 0.31, 0.0]),
    "key": np.array([0.7, 0.0, 0.714]),
    "color": np.array([0.0, 1.0, 0.0]),
}


class StubModel:
    def handle_word_cases(self, word):
        return word.lower()

    def encode_sentence_or_word(self, text):
        return VECTORS.get(text)


def operation(operation_id, param):
    return OperationProperties(
        operation_id=operation_id,
        endpoint_path=f"/{operation_id}",
        http_method="get",
        parameters={
            (param, "query"): ParameterProperties(
                name=param, in_value="query", schema=SchemaProperties(type="string")
            )
        },
    )


class SimilarityEdgeTests(unittest.TestCase):
    def setUp(self):
        self.comparator = OperationDependencyComparator(StubModel())

    def compare(self, param1, param2):
        similar, tentative = self.comparator.compare_cosine(
            operation("a", param1), operation("b", param2)
        )
        similar = {p: [sv.dependent_val for sv in svs] for p, svs in similar.items()}
        tentative = [(p, sv.dependent_val) for p, sv in tentative]
        return similar, tentative

    def test_matches_above_the_threshold_are_similar(self):
        self.assertEqual(
            self.compare("id", "identifier"),
            ({("id", "query"): [("identifier", "query")]}, []),
        )

    def test_matches_between_floor_and_threshold_are_tentative(self):
        self.assertEqual(
            self.compare("id", "key"), ({}, [(("id", "query"), ("key", "query"))])
        )

    def test_matches_below_the_floor_are_dropped(self):
        self.assertEqual(self.compare("id", "color"), ({}, []))

    def test_unmatched_operations_fall_back_to_tentative_edges(self):
        graph = OperationGraph(
            "unused", "probe", SimpleNamespace(config=Mock()), Mock()
        )
        graph.dependency_comparator = self.comparator
        operations = {
            "a": operation("a", "id"),
            "b": operation("b", "key"),
            "c": operation("c", "color"),
        }
        for properties in operations.values():
            graph.add_operation_node(properties)
        graph.determine_dependencies(operations)

        def destinations(op_id):
            return [e.destination.operation_id for e in graph.operation_nodes[op_id].outgoing_edges]

        # a and b fall back to each other; c has no match at or above the floor.
        self.assertEqual(destinations("a"), ["b"])
        self.assertEqual(destinations("b"), ["a"])
        self.assertEqual(destinations("c"), [])
        # Promoted edges are graph edges, so the DEPENDENCY data source is offered.
        self.assertEqual(
            {(e.source.operation_id, e.destination.operation_id) for e in graph.operation_edges},
            {("a", "b"), ("b", "a")},
        )
        self.assertIn("DEPENDENCY", DataSourceAgent(graph).available_data_sources)


if __name__ == "__main__":
    unittest.main()
