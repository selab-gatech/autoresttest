import copy
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

import requests
import pickle
import os

from autoresttest.config import Config
from autoresttest.models import (
    OperationProperties,
    ParameterKey,
    ParameterProperties,
    RequestData,
    RequestRequirements,
    RequestResponse,
    SchemaProperties,
)

from autoresttest.utils import (
    remove_nulls,
    split_parameter_values,
    get_params,
    get_request_body_params,
    get_param_combinations,
    get_required_params,
    get_body_object_combinations,
    dispatch_request,
    get_accept_header,
)
from autoresttest.llm import NaiveValueGenerator, SmartValueGenerator

if TYPE_CHECKING:
    from .generate_graph import OperationGraph, OperationNode, OperationEdge



@dataclass
class StatusCode:
    status_code: int
    count: int
    requests_and_responses: List[RequestResponse]


class RequestGenerator:
    def __init__(self, operation_graph: "OperationGraph", api_url: str, is_naive=True):
        self.operation_graph: "OperationGraph" = operation_graph
        self.config = operation_graph.config
        self.api_url = api_url
        self.status_codes: Dict[int, StatusCode] = (
            {}
        )  # dictionary to track status code occurrences
        self.requests_generated = 0  # Initialize the request count
        self.successful_query_data = []  # List to store successful query data
        self.is_naive = is_naive
        self.responses: Dict[str, RequestResponse] = (
            {}
        )  # Dictionary to store responses for each operation_id
        self.allowed_retries = 1

    @staticmethod
    def generate_naive_values(
        parameters: Dict[ParameterKey, ParameterProperties],
        request_body: Dict[str, SchemaProperties] | None,
    ):
        value_generator = NaiveValueGenerator(
            parameters=parameters, request_body=request_body
        )
        return value_generator.generate_parameters() if parameters else None, (
            value_generator.generate_request_body() if request_body else None
        )

    @staticmethod
    def generate_smart_values(
        operation_properties: OperationProperties,
        requirements: RequestRequirements | None = None,
        config: Config | None = None,
    ):
        """
        Generate smart values for parameters and request body using LLMs
        :param operation_properties: Dictionary mapping of operation properties
        :param requirements: RequestRequirements object that contains any parameters or request body requirements
        :return: a tuple of the generated parameters and request body
        """
        value_generator = SmartValueGenerator(
            operation_properties=operation_properties,
            requirements=requirements,
            config=config,
        )
        return (
            value_generator.generate_parameters(),
            value_generator.generate_request_body(),
        )

    def make_request_data(
        self,
        operation_properties: OperationProperties,
        requirements: RequestRequirements | None = None,
    ) -> RequestData:
        """
        Process the operation properties by preparing request data for queries with mapping values to parameters and request body
        """
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method.lower()

        if self.is_naive:
            parameters, request_body = self.generate_naive_values(
                operation_properties.parameters, operation_properties.request_body
            )
        else:
            parameters, request_body = self.generate_smart_values(
                operation_properties=operation_properties,
                requirements=requirements,
                config=self.config,
            )

        return RequestData(
            endpoint_path=endpoint_path,
            http_method=http_method,
            parameters=parameters,
            request_body=request_body,
            operation_properties=operation_properties,
            requirements=requirements,
        )

    def make_request_retry_data(
        self, request_data: RequestData, response: requests.Response
    ) -> RequestData:
        if self.is_naive:
            return request_data
        else:
            requirements = request_data.requirements
            value_generator = SmartValueGenerator(
                operation_properties=request_data.operation_properties,
                requirements=requirements,
                temperature=self.config.strict_temperature,
                config=self.config,
            )
            parameters, request_body = value_generator.generate_retry_parameters(
                request_data, response
            ), value_generator.generate_retry_request_body(request_data, response)
            return RequestData(
                endpoint_path=request_data.endpoint_path,
                http_method=request_data.http_method,
                parameters=parameters,
                request_body=request_body,
                operation_properties=request_data.operation_properties,
                requirements=requirements,
            )

    def _handle_retry(
        self,
        request_data: RequestData,
        response: requests.Response,
        curr_retry: int,
        permitted_retries: int = 1,
    ) -> Optional[RequestResponse]:
        retry_data = self.make_request_retry_data(request_data, response)
        retry_response = self.send_operation_request(
            retry_data,
            retry_nums=curr_retry + 1,
            allow_retry=True,
            permitted_retries=permitted_retries,
        )
        return retry_response

    def _determine_auth_mappings(
        self, request_val: Any, auth_parameters: Dict, auth_mappings
    ):
        if isinstance(request_val, dict):
            for item, properties in request_val.items():
                item_name = item[0] if isinstance(item, tuple) else item
                if auth_parameters.get("username") == item_name:
                    auth_mappings["username"] = properties
                elif auth_parameters.get("password") == item_name:
                    auth_mappings["password"] = properties
                else:
                    self._determine_auth_mappings(
                        properties, auth_parameters, auth_mappings
                    )
        elif isinstance(request_val, list):
            for item in request_val:
                self._determine_auth_mappings(item, auth_parameters, auth_mappings)

    def _determine_if_valid_auth(self, param_auth):
        if not param_auth or not isinstance(param_auth, dict):
            return False
        return (
            param_auth.get("username")
            and param_auth.get("username") != "None"
            and param_auth.get("password")
            and param_auth.get("password") != "None"
        )

    def get_auth_info(self, operation_node: "OperationNode", auth_attempts: int = 3):
        operation_properties = operation_node.operation_properties
        value_generator = SmartValueGenerator(
            operation_properties=operation_properties, config=self.config
        )

        auth_parameters = value_generator.determine_auth_params()
        # print(f"Auth parameters for operation {operation_node.operation_id}: {auth_parameters}")
        if not auth_parameters or auth_parameters == "None":
            return []
        query_param_auth = auth_parameters.get("query_parameters")
        body_param_auth = auth_parameters.get("body_parameters")

        print(
            "Attempting to get auth info for operation: ", operation_node.operation_id
        )

        token_info = []
        if query_param_auth and self._determine_if_valid_auth(query_param_auth):
            self.determine_tokens(operation_node, query_param_auth, token_info)
        elif body_param_auth and self._determine_if_valid_auth(body_param_auth):
            self.determine_tokens(operation_node, body_param_auth, token_info, True)
        return token_info

    def _decompose_body_prop_mappings(self, bodies: List):
        body_mappings = {}
        for body in bodies:
            if isinstance(body, dict):
                for key, value in body.items():
                    if key not in body_mappings:
                        body_mappings[key] = []
                    body_mappings[key].append(value)
        return body_mappings

    def determine_tokens(
        self,
        operation_node: "OperationNode",
        param_auth_info,
        token_info,
        is_body=False,
    ):
        param_q_table = {"params": {}, "body": {}}
        params = get_param_combinations(
            operation_node.operation_properties.parameters, config=self.config
        )
        if not is_body:
            for param in params:
                # param is a tuple of ParameterKey tuples; p[0] extracts the parameter name
                param_names = {p[0] for p in param}
                if set(param_auth_info.values()).issubset(param_names):
                    param_q_table["params"][param] = 0
        else:
            param_q_table["params"] = {param: 0 for param in params}
        select_mime = (
            list(operation_node.operation_properties.request_body.keys())[0]
            if operation_node.operation_properties.request_body
            else None
        )
        body_properties: List = []
        if select_mime and operation_node.operation_properties.request_body:
            body_properties = get_body_object_combinations(
                operation_node.operation_properties.request_body[select_mime],
                config=self.config,
            )
        if is_body:
            for body in body_properties:
                if set(param_auth_info.values()).issubset(set(body)):
                    param_q_table["body"][body] = 0
        else:
            param_q_table["body"] = {key: 0 for key in body_properties}

        value_generator = SmartValueGenerator(
            operation_properties=operation_node.operation_properties, config=self.config
        )
        failed_responses = []
        for i in range(3):
            response = self.create_and_send_request(
                operation_node, allow_retry=True, permitted_retries=1
            )
            if response is not None and response.response and not response.response.ok:
                failed_responses.append(response)
        if failed_responses:
            parameter_mappings, request_body_mappings = (
                value_generator.generate_informed_value_agent_params(
                    num_values=10, responses=failed_responses
                ),
                value_generator.generate_informed_value_agent_body(
                    num_values=10, responses=failed_responses
                ),
            )
        else:
            parameter_mappings, request_body_mappings = (
                value_generator.generate_value_agent_params(num_values=10),
                value_generator.generate_value_agent_body(num_values=10),
            )

        request_body_mappings = (
            self._decompose_body_prop_mappings(request_body_mappings[select_mime])
            if select_mime and select_mime in request_body_mappings
            else {}
        )

        param_q_table["params"]["None"] = 0
        param_q_table["body"]["None"] = 0

        curr_tokens = []
        start_time = time.time()
        while time.time() - start_time < 30 and len(curr_tokens) < 5:
            # get best action
            required_params = get_required_params(
                operation_node.operation_properties.parameters
            )
            best_params = (None, -np.inf)
            if param_q_table["params"] and required_params:
                for params, score in param_q_table["params"].items():
                    if (
                        params
                        and required_params.issubset(set(params))
                        and score > best_params[1]
                    ):
                        best_params = (params, score)
                best_params = best_params[0]
            else:
                best_params = (
                    max(param_q_table["params"].items(), key=lambda x: x[1])[0]
                    if param_q_table["params"]
                    else None
                )

            # required_body = get_required_params(operation_node.operation_properties.request_body[select_mime].properties)
            best_body = (
                max(param_q_table["body"].items(), key=lambda x: x[1])[0]
                if param_q_table["body"]
                else None
            )

            if best_params == "None":
                best_params = None
            if best_body == "None":
                best_body = None

            parameters = {}
            request_body = {}
            if best_params:
                for param in best_params:
                    parameters[param] = (
                        random.choice(parameter_mappings[param])
                        if param in parameter_mappings
                        else None
                    )
            if best_body and select_mime:
                request_body[select_mime] = {}
                for prop in best_body:
                    request_body[select_mime][prop] = (
                        random.choice(request_body_mappings[prop])
                        if prop in request_body_mappings
                        else None
                    )

            response = self.send_operation_request(
                RequestData(
                    endpoint_path=operation_node.operation_properties.endpoint_path,
                    http_method=operation_node.operation_properties.http_method,
                    parameters=parameters,
                    request_body=request_body,
                    operation_properties=operation_node.operation_properties,
                ),
                allow_retry=False,
                permitted_retries=0,
            )

            if best_params is None:
                best_params = "None"
            if best_body is None:
                best_body = "None"

            if response and response.response.ok:
                auth_mappings = {}
                self._determine_auth_mappings(
                    response.request.parameters, param_auth_info, auth_mappings
                )
                if len(auth_mappings) == 2:
                    curr_tokens.append(auth_mappings)
                auth_mappings = {}
                self._determine_auth_mappings(
                    response.request.request_body, param_auth_info, auth_mappings
                )
                if len(auth_mappings) == 2:
                    curr_tokens.append(auth_mappings)
                param_q_table["params"][best_params] += 2
                param_q_table["body"][best_body] += 2
            else:
                param_q_table["params"][best_params] -= 1
                param_q_table["body"][best_body] -= 1

        if curr_tokens:
            for token in curr_tokens:
                if token not in token_info:
                    token_info.append(token)

    def send_operation_request(
        self,
        request_data: RequestData,
        retry_nums: int = 0,
        allow_retry: bool = False,
        permitted_retries: int = 1,
    ) -> Optional[RequestResponse]:
        """
        Send the operation request to the API
        :param request_data:
        :param retry_nums:
        :param allow_retry:
        :return:
        """
        endpoint_path = request_data.endpoint_path
        http_method = request_data.http_method
        parameters = request_data.parameters
        request_body = request_data.request_body
        processed_parameters = copy.deepcopy(parameters) or {}

        path_params, query_params, header_params, cookie_params = (
            split_parameter_values(
                request_data.operation_properties.parameters, processed_parameters
            )
        )

        for name, value in path_params.items():
            endpoint_path = endpoint_path.replace("{" + name + "}", str(value))

        try:
            select_method = getattr(
                requests, http_method
            )  # selects correct http method
            full_url = f"{self.api_url}{endpoint_path}"

            # Merge custom headers from config
            merged_headers = header_params.copy()
            merged_headers.update(self.config.static_headers)

            accept_header = get_accept_header(
                request_data.operation_properties.responses
            )
            response = dispatch_request(
                select_method=select_method,
                full_url=full_url,
                params=query_params,
                body=request_body,
                header=merged_headers,
                cookies=cookie_params,
                accept=accept_header,
                config=self.config,
            )
            if response is not None:
                if not response.ok and retry_nums < permitted_retries and allow_retry:
                    return self._handle_retry(
                        request_data, response, retry_nums, permitted_retries
                    )
                else:
                    return RequestResponse(
                        request=request_data,
                        response=response,
                        response_text=response.text,
                    )
            return None
        except requests.exceptions.RequestException as err:
            print(f"Request exception due to error: {err}")
            print(f"Endpoint Path: {endpoint_path}")
            print(f"Params: {parameters}")
            print(f"Request Body: {request_body}")
            return None
        except Exception as err:
            print(f"Unexpected error due to: {err}")
            print(f"Endpoint Path: {endpoint_path}")
            print(f"Params: {parameters}")
            print(f"Request Body: {request_body}")
            return None

    def _validate_value_mappings(
        self,
        curr_node: "OperationNode",
        parameter_mappings: Dict,
        req_param_mappings: Dict[ParameterKey, List[Any]],
        req_body_mappings: Dict[str, List[Any]],
        occurrences: Dict,
    ):
        operation_id = curr_node.operation_id
        if operation_id not in parameter_mappings:
            parameter_mappings[operation_id] = {"params": {}, "body": {}}

        operation_params = get_params(curr_node.operation_properties.parameters)
        # operation_body_params = get_request_body_params(curr_node.operation_properties.request_body)

        # take the reverse of mappings since API outputs least likely values first
        req_param_mappings = (
            {k: v for k, v in reversed(list(req_param_mappings.items()))}
            if req_param_mappings
            else {}
        )
        req_body_mappings = (
            {k: v for k, v in reversed(list(req_body_mappings.items()))}
            if req_body_mappings
            else {}
        )

        if req_param_mappings:
            for parameter, values in req_param_mappings.items():
                for value in values:
                    if parameter not in parameter_mappings[operation_id]["params"]:
                        parameter_mappings[operation_id]["params"][parameter] = []
                    if (
                        parameter in operation_params
                        and len(parameter_mappings[operation_id]["params"][parameter])
                        < 10
                    ):
                        parameter_mappings[operation_id]["params"][parameter].append(
                            [value, 0]
                        )
                        occurrences[parameter] = occurrences.get(parameter, 0) + 1

        if req_body_mappings:
            for mime, mime_params in req_body_mappings.items():
                if mime not in parameter_mappings[operation_id]["body"]:
                    parameter_mappings[operation_id]["body"][mime] = []
                for mime_param in mime_params:
                    num_bodies = len(parameter_mappings[operation_id]["body"][mime])
                    if num_bodies < 10:
                        parameter_mappings[operation_id]["body"][mime].append(
                            [mime_param, 0]
                        )
                        occurrences[mime] = occurrences.get(mime, 0) + 1

                # body_params = get_nested_obj_mappings(mime_params) # handler works for getting first obj values same as get_request_body_params
                # for body_param, value in body_params.items():
                #    if body_param not in parameter_mappings[operation_id]['body'][mime]:
                #        parameter_mappings[operation_id]['body'][mime][body_param] = {}
                #    if body_param in operation_body_params[mime] and len(parameter_mappings[operation_id]['body'][mime][body_param]) < 10:
                #        parameter_mappings[operation_id]['body'][mime][body_param][value] = 0
                #        occurrences[body_param] = occurrences.get(body_param, 0) + 1

    def value_depth_traversal(
        self,
        curr_node: "OperationNode",
        parameter_mappings: Dict,
        responses: Dict[str, List],
        visited: Set,
    ):
        visited.add(curr_node.operation_id)

        # Pre-initialize Q-table entry to ensure it exists even if later steps fail
        if curr_node.operation_id not in parameter_mappings:
            parameter_mappings[curr_node.operation_id] = {"params": {}, "body": {}}

        for edge in curr_node.outgoing_edges:
            if edge.destination.operation_id not in visited:
                self.value_depth_traversal(
                    edge.destination, parameter_mappings, responses, visited
                )

        print("Building value table generation for operation: ", curr_node.operation_id)

        occurrences = {}
        # self.handle_dependent_values(curr_node, dependent_responses, occurrences, parameter_mappings, responses)

        desired_size = 10
        lowest_occurrences = min(occurrences.values()) if occurrences else 0
        if lowest_occurrences < desired_size:
            possible_responses = []
            # Use two samples with two retries each to gather server responses for augmenting value generation
            for _ in range(2):
                response = self.create_and_send_request(
                    curr_node, allow_retry=True, permitted_retries=2
                )
                if response is not None:
                    possible_responses.append(response)

            value_generator = SmartValueGenerator(
                operation_properties=curr_node.operation_properties, config=self.config
            )
            if possible_responses:
                parameters, request_body = (
                    value_generator.generate_informed_value_agent_params(
                        num_values=desired_size - lowest_occurrences,
                        responses=possible_responses,
                    ),
                    value_generator.generate_informed_value_agent_body(
                        num_values=desired_size - lowest_occurrences,
                        responses=possible_responses,
                    ),
                )
            else:
                parameters, request_body = value_generator.generate_value_agent_params(
                    num_values=desired_size - lowest_occurrences
                ), value_generator.generate_value_agent_body(
                    num_values=desired_size - lowest_occurrences
                )
            self._validate_value_mappings(
                curr_node, parameter_mappings, parameters, request_body, occurrences
            )

        print(
            "Completed value table generation for operation: ", curr_node.operation_id
        )

    def generate_value_tables_parallel(
        self,
        operation_nodes: Dict[str, "OperationNode"],
        parameter_mappings: Dict,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """Generate value tables for all operations in parallel using a thread pool."""
        # parameter_mappings corresponds to q_table
        visited_lock = threading.Lock()
        mappings_lock = threading.Lock()
        progress_lock = threading.Lock()
        visited: Set[str] = set()
        completed_count = 0

        def process_operation(operation_node: "OperationNode") -> str:
            nonlocal completed_count
            operation_id = operation_node.operation_id

            # Check if already processed (with lock)
            with visited_lock:
                if operation_id in visited:
                    return operation_id
                visited.add(operation_id)

            # Pre-initialize Q-table entry to ensure it exists even if later steps fail
            with mappings_lock:
                if operation_id not in parameter_mappings:
                    parameter_mappings[operation_id] = {"params": {}, "body": {}}

            # Generate values (I/O bound - HTTP + LLM calls)
            occurrences: Dict[str, int] = {}
            desired_size = 10
            lowest_occurrences = min(occurrences.values()) if occurrences else 0

            if lowest_occurrences < desired_size:
                possible_responses: List[RequestResponse] = []
                for _ in range(2):
                    response = self.create_and_send_request(
                        operation_node, allow_retry=True, permitted_retries=2
                    )
                    if response is not None:
                        possible_responses.append(response)

                value_generator = SmartValueGenerator(
                    operation_properties=operation_node.operation_properties,
                    config=self.config,
                )
                if possible_responses:
                    parameters, request_body = (
                        value_generator.generate_informed_value_agent_params(
                            num_values=desired_size - lowest_occurrences,
                            responses=possible_responses,
                        ),
                        value_generator.generate_informed_value_agent_body(
                            num_values=desired_size - lowest_occurrences,
                            responses=possible_responses,
                        ),
                    )
                else:
                    parameters, request_body = (
                        value_generator.generate_value_agent_params(
                            num_values=desired_size - lowest_occurrences
                        ),
                        value_generator.generate_value_agent_body(
                            num_values=desired_size - lowest_occurrences
                        ),
                    )

                # Thread-safe update to parameter_mappings
                with mappings_lock:
                    self._validate_value_mappings(
                        operation_node,
                        parameter_mappings,
                        parameters,
                        request_body,
                        occurrences,
                    )

            return operation_id

        # Submit all operations to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_operation, node): op_id
                for op_id, node in operation_nodes.items()
            }

            for future in as_completed(futures):
                op_id = futures[future]
                try:
                    result_op_id = future.result()
                    with progress_lock:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(result_op_id, completed_count)
                except Exception:
                    # Still increment count on error to avoid stuck progress
                    with progress_lock:
                        completed_count += 1
                        if progress_callback:
                            progress_callback(op_id, completed_count)

    def create_and_send_request(
        self,
        curr_node: "OperationNode",
        requirement: RequestRequirements | None = None,
        allow_retry: bool = False,
        permitted_retries: int = 1,
    ) -> Optional[RequestResponse]:
        request_data: RequestData = self.make_request_data(
            curr_node.operation_properties, requirement
        )
        response: Optional[RequestResponse] = self.send_operation_request(
            request_data, allow_retry=allow_retry, permitted_retries=permitted_retries
        )
        return response


if __name__ == "__main__":
    pass
