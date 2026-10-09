import logging
import os
from collections.abc import Sequence

import numpy as np
from gensim.downloader import load

from scipy.spatial.distance import cosine

from autoresttest.models import (
    OperationProperties,
    ParameterKey,
    SchemaProperties,
    SimilarityValue,
)

from autoresttest.utils import EmbeddingModel


class OperationDependencyComparator:
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self.threshold = 0.8
        # Matches below the threshold but at or above this floor become tentative
        # (fallback) edges for operations that have no match above the threshold.
        self.tentative_floor = 0.5

    def get_parameter_list(
        self, operation: OperationProperties
    ) -> list[tuple[str, ParameterKey, str]]:
        """Returns a list of the parameter name, its ParmaeterKey (name, location), and the location"""
        if not operation.parameters:
            return []
        parameter_list: list[tuple[str, ParameterKey, str]] = []
        for parameter_key, parameter_details in operation.parameters.items():
            processed_name = self.model.handle_word_cases(parameter_details.name)
            location = parameter_key[1] if isinstance(parameter_key, tuple) else None
            parameter_list.append(
                (
                    processed_name,
                    parameter_key,
                    location or parameter_details.in_value or "query",
                )
            )
        return parameter_list

    def handle_response_params(
        self, response: SchemaProperties, response_params: list[dict[str, str]]
    ) -> None:
        """In-place adds to a list where each item is a mapping of the processed item to its original name in the response spec."""
        if response.properties:
            for item, item_details in response.properties.items():
                if {self.model.handle_word_cases(item): item} not in response_params:
                    response_params.append({self.model.handle_word_cases(item): item})
                self.handle_response_params(item_details, response_params)
        elif response.items:
            self.handle_response_params(response.items, response_params)
        else:
            return

    def handle_body_params(self, body: SchemaProperties) -> list[tuple[str, str]]:
        """Returns a tuple of the processed body parameter name and its original name."""
        object_params: list[tuple[str, str]] = []
        if body.properties:
            object_params = [
                (self.model.handle_word_cases(item), item)
                for item, item_details in body.properties.items()
            ]
        elif body.items:
            object_params = self.handle_body_params(body.items)
        return object_params

    def get_request_body_list(
        self, operation: OperationProperties
    ) -> list[tuple[str, str, str]]:
        if operation.request_body is None:
            return []
        request_body_list = []
        for (
            request_body_type,
            request_body_properties,
        ) in operation.request_body.items():
            request_body_list += [
                (processed, item, "body")
                for processed, item in self.handle_body_params(request_body_properties)
            ]
        return request_body_list

    def get_response_list(
        self, operation: OperationProperties
    ) -> list[tuple[str, str, str]]:
        if operation.responses is None:
            return []
        response_list = []
        for status_code, response_properties in operation.responses.items():
            if status_code and status_code[0] == "2" and response_properties.content:
                for response, response_details in response_properties.content.items():
                    curr_responses = []
                    self.handle_response_params(response_details, curr_responses)
                    response_list += [
                        (processed, item, "response")
                        for processed_item in curr_responses
                        for processed, item in processed_item.items()
                    ]
        return response_list

    def cosine_similarity(
        self,
        operation1_vals: Sequence[tuple[str, ParameterKey | str, str]],
        operation2_vals: Sequence[tuple[str, ParameterKey | str, str]],
    ) -> dict[str | ParameterKey, list[SimilarityValue]]:
        """
        Returns parameters or body properties (str or ParameterKey) that might map to parameters or body properties or responses (str or ParameterKey) in other operations.
        """
        param_param_similarity: dict[str | ParameterKey, list[SimilarityValue]] = {}
        for processed_parameter, parameter_key, parameter_loc in operation1_vals:
            param_param_similarity.setdefault(parameter_key, [])
            for processed_dependency, dependency_key, dependency_loc in operation2_vals:
                param_embedding = self.model.encode_sentence_or_word(
                    processed_parameter
                )
                dependency_embedding = self.model.encode_sentence_or_word(
                    processed_dependency
                )
                if param_embedding is not None and dependency_embedding is not None:
                    similarity: float = 1.0 - float(
                        cosine(param_embedding, dependency_embedding)
                    )
                    param_param_similarity[parameter_key].append(
                        SimilarityValue(
                            dependent_val=dependency_key,
                            in_value=f"{parameter_loc} to {dependency_loc}",
                            similarity=similarity,
                        )
                    )

        return param_param_similarity

    def compare_cosine(
        self, operation1: OperationProperties, operation2: OperationProperties
    ) -> tuple[
        dict[str | ParameterKey, list[SimilarityValue]],
        list[tuple[str | ParameterKey, SimilarityValue]],
    ]:
        parameter_matchings: dict[str | ParameterKey, list[SimilarityValue]] = {}
        similar_parameters: dict[str | ParameterKey, list[SimilarityValue]] = {}
        next_most_similar_parameters: list[
            tuple[str | ParameterKey, SimilarityValue]
        ] = []

        operation1_parameters: list[tuple[str, ParameterKey, str]] = (
            self.get_parameter_list(operation1)
        )
        operation1_body: list[tuple[str, str, str]] = self.get_request_body_list(
            operation1
        )
        operation2_parameters: list[tuple[str, ParameterKey, str]] = (
            self.get_parameter_list(operation2)
        )
        operation2_body: list[tuple[str, str, str]] = self.get_request_body_list(
            operation2
        )
        operation2_responses: list[tuple[str, str, str]] = self.get_response_list(
            operation2
        )

        # Each parameter is the (processed_name, parameter key (name, location), and location)
        # Each body is the (processed body property, original body property, and "body")
        # Each response is the (processed response property, original reponse property, and "body")

        if operation1.parameters:
            if operation2.parameters:
                parameter_matchings = self.cosine_similarity(
                    operation1_parameters, operation2_parameters
                )
            if operation2.request_body:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_parameters, operation2_body
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.responses:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_parameters, operation2_responses
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)

        if operation1.request_body:
            if operation2.parameters:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_body, operation2_parameters
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.request_body:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_body, operation2_body
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.responses:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_body, operation2_responses
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)

        for parameter, similarities in parameter_matchings.items():
            for similarity in similarities:
                # Only parameters with a match above the threshold get an entry, so an
                # operation pair without one falls back to a tentative edge.
                if similarity.similarity > self.threshold:
                    similar_parameters.setdefault(parameter, []).append(similarity)
                elif similarity.similarity >= self.tentative_floor:
                    next_most_similar_parameters.append((parameter, similarity))

        return similar_parameters, next_most_similar_parameters
