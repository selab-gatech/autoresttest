import heapq
import logging

from .request_generator import RequestGenerator
from .similarity_comparator import OperationDependencyComparator

from autoresttest.models import OperationProperties, SimilarityValue, ParameterKey
from autoresttest.specification import SpecificationParser
from autoresttest.utils import EmbeddingModel


class OperationNode:
    def __init__(self, operation_properties: OperationProperties):
        self.operation_id = operation_properties.operation_id
        self.operation_properties: OperationProperties = operation_properties
        self.outgoing_edges: list[OperationEdge] = []
        self.tentative_edges: list[OperationEdge] = []


class OperationEdge:
    def __init__(
        self,
        source: OperationNode,
        destination: OperationNode,
        similar_parameters: dict[str | ParameterKey, list[SimilarityValue]],
    ):
        if similar_parameters is None:
            similar_parameters = {}
        self.source: OperationNode = source
        self.destination: OperationNode = destination
        self.similar_parameters: dict[str | ParameterKey, list[SimilarityValue]] = (
            similar_parameters  # have parameters as the key (similarity value has response param and in_value)
        )


class OperationGraph:
    def __init__(
        self,
        spec_path,
        spec_name,
        spec_parser: SpecificationParser,
        embedding_model,
    ):
        self.spec_path = spec_path
        self.spec_name = spec_name
        self.spec_parser = spec_parser
        self.config = spec_parser.config
        self.request_generator: RequestGenerator | None = None
        self.operation_nodes: dict[str, OperationNode] = {}
        self.operation_edges: list[OperationEdge] = []
        self.next_most_similar_count = 3
        self.embedding_model: EmbeddingModel = embedding_model
        self.dependency_comparator = OperationDependencyComparator(
            model=embedding_model
        )

    def print_graph(self):
        for operation_id, operation_node in self.operation_nodes.items():
            print("=====================================")
            print(f"Operation: {operation_id}")
            # print(f"Operation Properties: {operation_node.operation_properties}")
            for edge in operation_node.outgoing_edges:
                print(
                    f"Edge: {edge.source.operation_id} -> {edge.destination.operation_id} with parameters: {edge.similar_parameters}"
                )
            for tentative_edge in operation_node.tentative_edges:
                print(
                    f"Tentative Edge: {tentative_edge.source.operation_id} -> {tentative_edge.destination.operation_id} with parameters: {tentative_edge.similar_parameters}"
                )
            print()
            print()

    def print_edges(self):
        for operation_edge in self.operation_edges:
            print(
                f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.similar_parameters}"
            )

    def add_operation_node(self, operation_properties: OperationProperties):
        self.operation_nodes[operation_properties.operation_id] = OperationNode(
            operation_properties
        )

    def assign_request_generator(self, request_generator: RequestGenerator):
        self.request_generator = request_generator

    def add_operation_edge(
        self,
        operation_id: str,
        dependent_operation_id: str,
        parameters: dict[str | ParameterKey, list[SimilarityValue]],
    ):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        if dependent_operation_id not in self.operation_nodes:
            raise ValueError(
                f"Dependent operation {dependent_operation_id} not found in the graph"
            )
        source_node = self.operation_nodes[operation_id]
        destination_node = self.operation_nodes[dependent_operation_id]
        edge = OperationEdge(
            source=source_node,
            destination=destination_node,
            similar_parameters=parameters,
        )
        self.operation_edges.append(edge)
        source_node.outgoing_edges.append(edge)
        # print(f"Added edge from {operation_id} to {dependent_operation_id} with parameters: {parameters}")

    def add_tentative_edge(
        self,
        operation_id: str,
        dependent_operation_id: str,
        next_closest_similarities: list[tuple[str | ParameterKey, SimilarityValue]],
    ):
        # TODO: Update tentative edge handling for lists
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        if dependent_operation_id not in self.operation_nodes:
            raise ValueError(
                f"Dependent operation {dependent_operation_id} not found in the graph"
            )
        source_node = self.operation_nodes[operation_id]
        destination_node = self.operation_nodes[dependent_operation_id]
        similar_parameters: dict[str | ParameterKey, list[SimilarityValue]] = {}
        # recall that next_closest_similarities is a list matching params in operation to params in dependent_operation
        for next_closest_similarity in next_closest_similarities:
            if next_closest_similarity[0] not in similar_parameters:
                similar_parameters[next_closest_similarity[0]] = []
            similar_parameters[next_closest_similarity[0]].append(
                next_closest_similarity[1]
            )

        edge = OperationEdge(
            source=source_node,
            destination=destination_node,
            similar_parameters=similar_parameters,
        )
        source_node.tentative_edges.append(edge)
        source_node.tentative_edges = heapq.nlargest(
            self.next_most_similar_count,
            [e for e in source_node.tentative_edges if e.similar_parameters],
            key=lambda x: max(
                sv.similarity for sv in next(iter(x.similar_parameters.values()))
            ),
        )  # small n so efficient

    def update_operation_dependencies(
        self,
        operation_id: str,
        dependent_operation_id: str,
        similar_parameters: dict[str | ParameterKey, list[SimilarityValue]],
        next_closest_similarities: list[tuple[str | ParameterKey, SimilarityValue]],
    ):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        if len(similar_parameters) > 0:
            self.add_operation_edge(
                operation_id, dependent_operation_id, similar_parameters
            )
        # Only add tentative edges if there are no similar parameters
        elif len(next_closest_similarities) > 0:
            self.add_tentative_edge(
                operation_id, dependent_operation_id, next_closest_similarities
            )

    def remove_edge(self, operation_id: str, dependent_operation_id: str):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        edge_to_remove = None
        for edge in source_node.outgoing_edges:
            if edge.destination.operation_id == dependent_operation_id:
                edge_to_remove = edge
                break
        if edge_to_remove:
            source_node.outgoing_edges.remove(edge_to_remove)
            self.operation_edges.remove(edge_to_remove)

    def determine_dependencies(
        self, operations: dict[str, OperationProperties]
    ) -> None:
        for operation_id, operation_properties in operations.items():
            for (
                dependent_operation_id,
                dependent_operation_properties,
            ) in operations.items():
                if operation_id == dependent_operation_id:
                    continue
                # Note: Matches parameters and request body properties of each operation with the parameters, request body properties, and response properties of another operation
                parameter_similarities, next_closest_similarities = (
                    self.dependency_comparator.compare_cosine(
                        operation_properties, dependent_operation_properties
                    )
                )
                self.update_operation_dependencies(
                    operation_id,
                    dependent_operation_id,
                    parameter_similarities,
                    next_closest_similarities,
                )
            if (
                self.operation_nodes[operation_id].tentative_edges
                and not self.operation_nodes[operation_id].outgoing_edges
            ):
                # Assign top tentative edges to outgoing edges if there are no similar parameters
                self.operation_nodes[operation_id].outgoing_edges = (
                    self.operation_nodes[operation_id].tentative_edges
                )

    def create_graph(self, auto_validate: bool = True) -> None:
        operations: dict[str, OperationProperties] = (
            self.spec_parser.parse_specification()
        )
        for operation_id, operation_properties in operations.items():
            self.add_operation_node(operation_properties)
        self.determine_dependencies(operations)


if __name__ == "__main__":
    from pathlib import Path

    # Get project root: generate_graph.py -> graph -> autoresttest -> src -> project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    spec_path = PROJECT_ROOT / "seawedfs.yaml"

    # Create required dependencies
    embedding_model = EmbeddingModel()
    spec_parser = SpecificationParser(spec_path=str(spec_path))

    operation_graph = OperationGraph(
        spec_path=str(spec_path),
        spec_name="seawedfs",
        spec_parser=spec_parser,
        embedding_model=embedding_model,
    )
    operation_graph.create_graph()
    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print("=====================================")
        print(f"Operation: {operation_id}")
        for edge in operation_node.outgoing_edges:
            print(
                f"Edge: {edge.source.operation_id} -> {edge.destination.operation_id} with parameters: {edge.similar_parameters}"
            )
        for tentative_edge in operation_node.tentative_edges:
            print(
                f"Tentative Edge: {tentative_edge.source.operation_id} -> {tentative_edge.destination.operation_id} with parameters: {tentative_edge.similar_parameters}"
            )
        print()
        print()

    # for operation_edge in operation_graph.operation_edges:
    #    print(f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.similar_parameters}")
