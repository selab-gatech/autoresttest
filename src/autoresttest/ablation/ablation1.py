import copy
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import requests
import shelve

import sys

# Get the root directory (two levels up)
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_directory)

from autoresttest.config import Config, get_config
from autoresttest.graph.generate_graph import OperationGraph
from autoresttest.specification import SpecificationParser
from autoresttest.models import OperationProperties
from autoresttest.agents import OperationAgent, HeaderAgent, ParameterAgent, ValueAgent, ValueAction, BodyObjAgent, \
    DataSourceAgent, DependencyAgent
from autoresttest.graph import RequestGenerator
from autoresttest.utils import construct_db_dir, construct_basic_token, get_body_params, \
    get_response_params, get_response_param_mappings, remove_nulls, encode_dictionary, EmbeddingModel, get_api_url, \
    dispatch_request
from autoresttest.llm import identify_generator, randomize_string, random_generator, randomize_object




class Ablation1:
    def __init__(self, operation_graph, alpha=0.1, gamma=0.9, epsilon=0.3, time_duration=600, mutation_rate=0.3):
        self.q_table = {}
        self.operation_graph: OperationGraph = operation_graph
        self.config = operation_graph.config
        self.api_url = operation_graph.request_generator.api_url
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
        self.operation_agent = OperationAgent(operation_graph, alpha, gamma, 0.7)
        self.header_agent = HeaderAgent(operation_graph, alpha, gamma, epsilon)
        self.parameter_agent = ParameterAgent(operation_graph, alpha, gamma, epsilon)
        self.value_agent = ValueAgent(operation_graph, alpha, gamma, epsilon)
        self.body_object_agent = BodyObjAgent(operation_graph, alpha, gamma, epsilon)
        self.data_source_agent = DataSourceAgent(operation_graph, alpha, gamma, 0.7)
        self.dependency_agent = DependencyAgent(operation_graph, alpha, gamma, epsilon)
        self.time_duration = time_duration
        self.responses = defaultdict(int)

        self.errors = {}
        self.unique_errors = {}
        self.successful_parameters = {}
        self.successful_bodies = {}
        self.successful_responses = {}
        self.successful_primitives = {}
        self.operation_response_counter = {}
        self._init_parameter_tracking()
        self._init_body_tracking()
        self._init_response_tracking()

    def print_q_tables(self):
        print("OPERATION Q-TABLE: ", self.operation_agent.q_table)
        print("HEADER Q-TABLE: ", self.header_agent.q_table)
        print("PARAMETER Q-TABLE: ", self.parameter_agent.q_table)
        print("VALUE Q-TABLE: ", self.value_agent.q_table)

    def get_mapping(self, select_params, select_values):
        if not select_params:
            return None
        return {param: select_values[param] for param in select_params if param in select_values}

    def get_mutated_value(self, param_type):
        if not param_type:
            return None
        avail_types = ["integer", "number", "string", "boolean", "array", "object"]
        avail_types.remove(param_type)
        return identify_generator(random.choice(avail_types))()

    # Gets a random value from the table of primative responses
    def assign_random_from_primitives(self, parameters, body, operation_id):
        possible_options = [val for key, val in self.successful_primitives.items() if key != operation_id]

        if parameters:
            for parameter in parameters:
                if random.random() < 0.7 and possible_options:
                    parameters[parameter] = random.choice(possible_options)

        if body:
            for mime, body_properties in body.items():
                if random.random() < 0.7 and possible_options:
                    body[mime] = random.choice(possible_options)

        return parameters, body

    # Get an object from responses that completely matches the body format
    def assign_random_complete_body(self, body, operation_id, complete_body_mappings):
        new_body = {}
        if body:
            for mime, body_properties in body.items():
                if operation_id in complete_body_mappings:
                    body_mappings = self._deconstruct_body(body_properties)
                    if body_mappings:
                        new_body[mime] = self._construct_body({prop: random_generator()() for prop in body_mappings},
                                                              operation_id, mime)
        return new_body

    def assign_random_from_successful(self, parameters, body, operation_id, complete_body_mappings):

        # DEPRECATED

        possible_options = []

        # Add all parameters with name: value
        for operation_idx, operation_parameters in self.successful_parameters.items():
            if operation_idx == operation_id:
                continue
            for parameter_name, parameter_values in operation_parameters.items():
                for parameter_value in parameter_values:
                    possible_options.append({"name": parameter_name, "value": parameter_value})

        for operation_idx, operation_body_parms in self.successful_bodies.items():
            if operation_idx == operation_id:
                continue
            for body_name, body_values in operation_body_parms.items():
                for body_value in body_values:
                    possible_options.append({"name": body_name, "value": body_value})

        for operation_idx, operation_responses in self.successful_responses.items():
            if operation_idx == operation_id:
                continue
            for response_name, response_values in operation_responses.items():
                for response_value in response_values:
                    possible_options.append({"name": response_name, "value": response_value})

        for operation_idx, operation_primitives in self.successful_primitives.items():
            if operation_idx == operation_id:
                continue
            #.extend(operation_primitives)

        if parameters:
            for parameter in parameters:
                if random.random() < 0.7 and possible_options:
                    parameters[parameter] = random.choice(possible_options)

        if body:
            for mime, body_properties in body.items():
                if random.random() < 0.3 and operation_id in complete_body_mappings: # Try to assign a random successful body object
                    body_mappings = self._deconstruct_body(body_properties)
                    if body_mappings:
                        new_obj = {}
                        for prop in body_mappings:
                            if random.random() < 0.5 and possible_options:
                                new_obj[prop] = random.choice(possible_options)
                            else:
                                new_obj[prop] = body_mappings[prop]
                        body[mime] = self._construct_body(new_obj, operation_id, mime)
                else:
                    if random.random() < 0.3 and operation_id in complete_body_mappings:
                        possible_objs = []
                        for dependent_operation, dependent_prop in complete_body_mappings[operation_id].items():
                            if dependent_operation in self.successful_bodies and dependent_prop in \
                                    self.successful_bodies[dependent_operation]:
                                possible_objs.extend(self.successful_bodies[dependent_operation][dependent_prop])
                        if possible_objs:
                            selected_obj = random.choice(possible_objs)
                            body[mime] = selected_obj
                    elif operation_id in complete_body_mappings:
                        body_mappings = self._deconstruct_body(body_properties)
                        if body_mappings:
                            possible_objs = []
                            for dependent_operation, dependent_prop in complete_body_mappings[operation_id].items():
                                if dependent_operation in self.successful_bodies and dependent_prop in self.successful_bodies[
                                    dependent_operation]:
                                    possible_objs.extend(self.successful_bodies[dependent_operation][dependent_prop])
                            if possible_objs:
                                selected_obj = random.choice(possible_objs)
                                new_obj = {}
                                response_mappings = self._deconstruct_body(selected_obj)
                                if response_mappings:
                                    for prop in body_mappings:
                                        if prop in response_mappings:
                                            new_obj[prop] = response_mappings[prop]
                                        else:
                                            new_obj[prop] = body_mappings[prop]
                                    body[mime] = self._construct_body(new_obj, operation_id, mime)

        return parameters, body

    def determine_complete_body_mappings(self):
        """
        Determine mapping between operations that consist of complete objects
        :return: A dictionary containing the connection between operations and the name of the property containing the complete object
        """
        complete_body_mappings = {}
        for operation1_id, operation1_node in self.operation_graph.operation_nodes.items():
            for operation2_id, operation2_node in self.operation_graph.operation_nodes.items():
                if operation1_id == operation2_id:
                    continue
                if operation1_node.operation_properties.request_body and operation2_node.operation_properties.responses:
                    for mime_type, body_properties in operation1_node.operation_properties.request_body.items():
                        if not body_properties.properties:
                            continue
                        for status_code, response_properties in operation2_node.operation_properties.responses.items():
                            if status_code and status_code[0] == "2" and response_properties.content:
                                for content_type, response_details in response_properties.content.items():
                                    response_params = {}
                                    get_response_param_mappings(response_details, response_params)
                                    for prop, val in response_params.items():
                                        if val.properties == body_properties.properties:
                                            complete_body_mappings.setdefault(operation1_id, {})[operation2_id] = prop
        return complete_body_mappings

    def mutate_values(self, operation_properties: OperationProperties, parameters, body, header):
        avail_medias = ["application/json", "application/x-www-form-urlencoded", "multipart/form-data", "text/plain"]
        avail_methods = ["get", "post", "put", "delete", "patch"]

        parameter_type_mutate_rate = 0.5
        body_mutate_rate = 0.3
        mutate_method = random.random() < 0
        mutate_media = random.random() < 0.02
        mutate_parameter_completely = random.random() < 0.05
        mutate_token = random.random() < 0.2

        specific_method = None
        if operation_properties.http_method and mutate_method and operation_properties.http_method.lower() in avail_methods:
            avail_methods.remove(operation_properties.http_method.lower())
            specific_method = random.choice(avail_methods)

        if mutate_token:
            random_token_params = {"username": randomize_string(), "password": randomize_string()}
            if self.config.enable_header_agent and header and random.random() < 0.5:
                header = None
            else:
                header = {"Authorization": construct_basic_token(random_token_params)}

        mutated_parameter_names = False
        if random.random() < mutate_parameter_completely:
            mutated_parameter_names = True
            parameters = {randomize_string(): random_generator()() for _ in range(random.randint(2,6))}

        if operation_properties.parameters and parameters and not mutate_parameter_completely:
            for parameter_name, parameter_properties in operation_properties.parameters.items():
                if parameter_name in parameters:
                    mutate_parameter_type = random.random() < parameter_type_mutate_rate
                    if parameter_properties.schema and mutate_parameter_type:
                        if parameter_properties.schema.type:
                            parameters[parameter_name] = self.get_mutated_value(parameter_properties.schema.type)
                        else:
                            parameters[parameter_name] = random_generator()()

                    if parameters[parameter_name] is None:
                        parameters.pop(parameter_name, None)

        if operation_properties.request_body and body:
            for mime, body_properties in operation_properties.request_body.items():
                mutate_body = random.random() < body_mutate_rate
                if mime in body and mutate_body:
                    if random.random() < 0.5 and body_properties.type:
                        body[mime] = self.get_mutated_value(body_properties.type)
                    else:
                        body[mime] = randomize_object()
                if mime in body and body[mime] is None:
                    body.pop(mime, None)

        if random.random() < mutate_media and body:
            for media in body.keys():
                avail_medias.remove(media)
            new_body = {random.choice(avail_medias): body.popitem()[1]}
            body = new_body

        return parameters, body, header, specific_method, mutated_parameter_names

    def send_operation(self, operation_properties: OperationProperties, parameters, body, header, specific_method=None):
        endpoint_path = operation_properties.endpoint_path
        http_method = specific_method if specific_method else operation_properties.http_method.lower()
        processed_parameters = copy.deepcopy(parameters)

        if processed_parameters:
            for parameter_name, parameter_properties in operation_properties.parameters.items():
                if parameter_properties.in_value == "path" and parameter_name in processed_parameters:
                    path_value = processed_parameters[parameter_name]
                    endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(path_value))
                    processed_parameters.pop(parameter_name, None)

        #self._test_send_operation(operation_properties, parameters, body, header, specific_method)

        try:
            select_method = getattr(requests, http_method)
            full_url = self.api_url + endpoint_path
            response = dispatch_request(
                select_method=select_method,
                full_url=full_url,
                params=processed_parameters,
                body=body,
                header=header,
                config=self.config,
            )
            return response
        except requests.exceptions.RequestException as err:
            print(f"Error with operation {operation_properties.operation_id}: {err}")
            return None
        except Exception as err:
            print(f"Unexpected error with operation {operation_properties.operation_id}: {err}")
            print("Parameters: ", processed_parameters)
            print("Body: ", body)
            return None

    def _test_send_operation(self, operation_properties, parameters, body, header, specific_method):
        print("=============================================")
        print("Operation ID: ", operation_properties.operation_id)
        print("Parameters: ", parameters)
        print("Body: ", body)
        print("Header: ", header)
        print("Specific Method: ", specific_method)
        print("=============================================")

    def determine_header_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code == 401:
            return -3
        elif status_code // 100 == 4:
            return -1
        elif status_code // 100 == 5:
            return -1
        elif status_code // 100 == 2:
            return 2
        else:
            return -3

    def determine_value_response_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code // 100 == 2:
            return 2
        elif status_code == 405:
            return -5
        elif status_code // 100 == 4:
            return -2
        elif status_code // 100 == 5:
            return -1
        else:
            return -5

    def determine_parameter_response_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code // 100 == 2:
            return 2
        elif status_code == 405:
            return -5
        elif status_code // 100 == 4:
            return -2
        elif status_code // 100 == 5:
            return -1
        else:
            return -5

    def determine_good_response_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code // 100 == 2:
            return 2
        elif status_code == 405:
            return -3
        elif status_code // 100 == 4:
            return -1
        elif status_code // 100 == 5:
            return -1
        else:
            return -5

    def determine_bad_response_reward(self, response):
        if response is None:
            return -10
        status_code = response.status_code
        if status_code == 405:
            return -10
        elif status_code == 401:
            return -3
        elif status_code // 100 == 4:
            return 1
        elif status_code // 100 == 5:
            return 2
        elif status_code // 100 == 2:
            return -1
        else:
            return -5

    def _init_parameter_tracking(self):
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.successful_parameters:
                self.successful_parameters[operation_id] = {}
            if operation_node.operation_properties.parameters:
                for parameter_name in operation_node.operation_properties.parameters.keys():
                    self.successful_parameters[operation_id][parameter_name] = []

    def _init_body_tracking(self):
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.successful_bodies:
                self.successful_bodies[operation_id] = {}
            if operation_node.operation_properties.request_body:
                for mime_type, body_properties in operation_node.operation_properties.request_body.items():
                    body_params = get_body_params(body_properties)
                    self.successful_bodies[operation_id] = {param: [] for param in body_params}

    def _init_response_tracking(self):
        for operation_id, operation_node in self.operation_graph.operation_nodes.items():
            if operation_id not in self.successful_responses:
                self.successful_responses[operation_id] = {}
            if operation_node.operation_properties.responses:
                for response_type, response_properties in operation_node.operation_properties.responses.items():
                    if response_properties.content:
                        for response, response_details in response_properties.content.items():
                            response_params = []
                            get_response_params(response_details, response_params)
                            self.successful_responses[operation_id] = {param: [] for param in response_params}

    def _construct_body_property(self, body_property, unconstructed_body):
        if body_property.properties or body_property.type == "object":
            return {prop: val for prop, val in unconstructed_body.items() if prop in body_property.properties}
        elif body_property.items or body_property.type == "array":
            return [self._construct_body_property(body_property.items, unconstructed_body)]
        else:
            return None

    def _construct_body(self, unconstructed_body, operation_id, mime_type):
        op_props = self.operation_graph.operation_nodes[operation_id].operation_properties
        for mime, body_properties in op_props.request_body.items():
            if mime == mime_type:
                return self._construct_body_property(body_properties, unconstructed_body)

    def _deconstruct_body(self, body):
        if body is None:
            return None
        if type(body) == dict:
            return {prop: val for prop, val in body.items()}
        elif type(body) == list:
            possible_length = len(body)
            if possible_length > 0:
                return self._deconstruct_body(body[random.randint(0, possible_length - 1)])
            else:
                return None
        else:
            return None

    def _deconstruct_response(self, response, response_mappings: Dict[str, List]):
        if response is None:
            return
        if type(response) == dict:
            for prop, val in response.items():
                if prop not in response_mappings:
                    response_mappings[prop] = []
                if val not in response_mappings[prop]:
                    response_mappings[prop].append(val)
                self._deconstruct_response(val, response_mappings)
        elif type(response) == list:
            for item in response:
                self._deconstruct_response(item, response_mappings)

    def generate_default_values(self, operation_id):
        default_assignments = {
            "integer": 1,
            "number": 1.0,
            "string": "default",
            "boolean": True,
            "array": ["default"],
            "object": {"default": 1}
        }

        def _safe_default(value_type):
            if not value_type:
                return random_generator()()
            if value_type in default_assignments:
                return default_assignments[value_type]
            generator = identify_generator(value_type)
            return generator() if callable(generator) else random_generator()()

        parameters = {}
        if self.operation_graph.operation_nodes[operation_id].operation_properties.parameters:
            for parameter_name, parameter_properties in self.operation_graph.operation_nodes[operation_id].operation_properties.parameters.items():
                if parameter_properties.schema:
                    value_type = getattr(parameter_properties.schema, "type", None)
                    if random.random() < 0.75 and value_type:
                        parameters[parameter_name] = _safe_default(value_type)
                    elif value_type:
                        generator = identify_generator(value_type)
                        parameters[parameter_name] = generator() if callable(generator) else random_generator()()
                    else:
                        parameters[parameter_name] = random_generator()()
        body = {}
        if self.operation_graph.operation_nodes[operation_id].operation_properties.request_body:
            for mime_type, body_properties in self.operation_graph.operation_nodes[operation_id].operation_properties.request_body.items():
                if not body_properties:
                    continue
                if body_properties.properties:
                    for prop in body_properties.properties:
                        prop_type = getattr(body_properties.properties[prop], "type", None)
                        if random.random() < 0.75 and prop_type:
                            body[mime_type] = {prop: _safe_default(prop_type)}
                        elif prop_type:
                            generator = identify_generator(prop_type)
                            body[mime_type] = {prop: generator() if callable(generator) else random_generator()()}
                        else:
                            body[mime_type] = {prop: random_generator()()}
                elif body_properties.items:
                    item_type = getattr(body_properties.items, "type", None)
                    if random.random() < 0.75 and item_type:
                        body[mime_type] = [_safe_default(item_type)]
                    elif item_type:
                        generator = identify_generator(item_type)
                        body[mime_type] = [generator() if callable(generator) else random_generator()()]
                    else:
                        body[mime_type] = [random_generator()()]
                else:
                    mime_type_val = getattr(body_properties, "type", None)
                    if random.random() < 0.75 and mime_type_val:
                        body[mime_type] = _safe_default(mime_type_val)
                    elif mime_type_val:
                        generator = identify_generator(mime_type_val)
                        body[mime_type] = generator() if callable(generator) else random_generator()()
                    else:
                        body[mime_type] = random_generator()()
        return parameters, body

    def select_exploration_agent(self, operation_id, start_time):

        # DEPRECATED: Using value decomposition idea for Q-value updating

        elapsed_time = time.time() - start_time

        if self.config.enable_header_agent:
            agent_options = ["PARAMETER & BODY", "DATA_SOURCE", "VALUE", "DEPENDENCY", "HEADER", "NONE", "ALL"]
        else:
            agent_options = ["PARAMETER & BODY", "DATA_SOURCE", "VALUE", "DEPENDENCY", "NONE", "ALL"]

        # Use exponential decay to allow for all agents to explore during initial time
        all_exploring_base_probability = 0.15
        # all_exploring_decay_rate = 1/150
        all_exploring_decay_rate = (-1 * 0.2 * self.time_duration) / (np.log(0.05)) # 0.1 is the desired probability at 20% of the time duration
        priority_exploring_space = 0.35
        all_exploring_probability = all_exploring_base_probability + (1 - all_exploring_base_probability) * np.exp(-all_exploring_decay_rate * elapsed_time)
        all_exploring_probability = min(all_exploring_probability, 1)
        remaining_probability = 1 - all_exploring_probability
        num_remaining = len(agent_options) - 1
        # Distribute some space for priority exploration
        other_event_probability = (1 - priority_exploring_space) * remaining_probability / num_remaining

        parameter_unexplored = self.parameter_agent.number_of_zeros(operation_id) + self.body_object_agent.number_of_zeros(operation_id)
        data_source_unexplored = self.data_source_agent.number_of_zeros(operation_id)
        value_unexplored = self.value_agent.number_of_zeros(operation_id)
        dependency_unexplored = self.dependency_agent.number_of_zeros(operation_id)

        baseline_probability = np.array([other_event_probability] * num_remaining + [all_exploring_probability], dtype=np.float64)
        unexplored_tables = np.array([parameter_unexplored, data_source_unexplored, value_unexplored, dependency_unexplored], dtype=np.float64)

        if np.sum(unexplored_tables) == 0:
            select_probabilities = baseline_probability / np.sum(baseline_probability)
        else:
            unexplored_tables /= np.sum(unexplored_tables)
            unexplored_tables *= 1.00 - np.sum(baseline_probability)
            select_probabilities = baseline_probability.copy()
            select_probabilities[:4] += unexplored_tables
            select_probabilities /= np.sum(select_probabilities) # Normalize for good measure

        exploring_agent = np.random.choice(agent_options, p=select_probabilities)
        return exploring_agent

    def execute_operations(self):
        start_time = time.time()

        complete_body_mappings = self.determine_complete_body_mappings()

        while time.time() - start_time < self.time_duration:

            operation_id = self.operation_agent.get_action()

            self.tui_output(start_time, operation_id)

            select_params = self.parameter_agent.get_action(operation_id)

            # Determine header
            if self.config.enable_header_agent:
                select_header = self.header_agent.get_action(operation_id)
            else:
                select_header = None

            data_source = self.data_source_agent.get_action(operation_id)

            # Determine value assignments
            parameter_dependencies, request_body_dependencies, unconstructed_body, select_values, dependency_type = None, None, {}, None, None
            if data_source == "DEFAULT":
                param_mappings, body_mappings = self.generate_default_values(operation_id)
                parameters = self.get_mapping(select_params.req_params, param_mappings) if select_params.req_params else None
                body = self.get_mapping([select_params.mime_type], body_mappings) if select_params.mime_type else None
            elif data_source == "DEPENDENCY":
                dependency_type, parameter_dependencies, request_body_dependencies = self.dependency_agent.get_action(operation_id, self)

                default_parameters, default_body = self.generate_default_values(operation_id)
                supplement_parameters = self.get_mapping(select_params.req_params, default_parameters) \
                    if select_params.req_params else None
                supplement_body = self.get_mapping([select_params.mime_type], default_body) \
                    if select_params.mime_type else None

                parameters = {}
                if select_params.req_params:
                    for parameter, dependency in parameter_dependencies.items():
                        if parameter in select_params.req_params:
                            if dependency["in_value"] == "params" and dependency["dependent_operation"] in self.successful_parameters and dependency["dependent_val"] in self.successful_parameters[dependency["dependent_operation"]]:
                                if self.successful_parameters[dependency["dependent_operation"]][dependency["dependent_val"]]:
                                    parameters[parameter] = random.choice(self.successful_parameters[dependency["dependent_operation"]][dependency["dependent_val"]])

                            elif dependency["in_value"] == "body" and dependency["dependent_operation"] in self.successful_bodies and dependency["dependent_val"] in self.successful_bodies[dependency["dependent_operation"]]:
                                if self.successful_bodies[dependency["dependent_operation"]][dependency["dependent_val"]]:
                                    parameters[parameter] = random.choice(self.successful_bodies[dependency["dependent_operation"]][dependency["dependent_val"]])

                            elif dependency["in_value"] == "response" and dependency["dependent_operation"] in self.successful_responses and dependency["dependent_val"] in self.successful_responses[dependency["dependent_operation"]]:
                                if self.successful_responses[dependency["dependent_operation"]][dependency["dependent_val"]]:
                                    parameters[parameter] = random.choice(self.successful_responses[dependency["dependent_operation"]][dependency["dependent_val"]])
                    for param in select_params.req_params:
                        if param not in parameters or not parameters[param]:
                            parameters[param] = supplement_parameters[param] if supplement_parameters and param in supplement_parameters else random_generator()()

                body = {}
                if select_params.mime_type and select_params.mime_type in self.operation_graph.operation_nodes[operation_id].operation_properties.request_body:
                    unconstructed_body = {}
                    possible_body_properties = get_body_params(self.operation_graph.operation_nodes[operation_id].operation_properties.request_body[select_params.mime_type])
                    for body_property, dependency in request_body_dependencies.items():
                        if body_property in possible_body_properties:
                            if dependency["in_value"] == "params" and dependency["dependent_operation"] in self.successful_parameters and dependency["dependent_val"] in self.successful_parameters[dependency["dependent_operation"]]:
                                if self.successful_parameters[dependency["dependent_operation"]][dependency["dependent_val"]]:
                                    unconstructed_body[body_property] = random.choice(self.successful_parameters[dependency["dependent_operation"]][dependency["dependent_val"]])

                            elif dependency["in_value"] == "body" and dependency["dependent_operation"] in self.successful_bodies and dependency["dependent_val"] in self.successful_bodies[dependency["dependent_operation"]]:
                                if self.successful_bodies[dependency["dependent_operation"]][dependency["dependent_val"]]:
                                    unconstructed_body[body_property] = random.choice(self.successful_bodies[dependency["dependent_operation"]][dependency["dependent_val"]])

                            elif dependency["in_value"] == "response" and dependency["dependent_operation"] in self.successful_responses and dependency["dependent_val"] in self.successful_responses[dependency["dependent_operation"]]:
                                if self.successful_responses[dependency["dependent_operation"]][dependency["dependent_val"]]:
                                    unconstructed_body[body_property] = random.choice(self.successful_responses[dependency["dependent_operation"]][dependency["dependent_val"]])

                    deconstructed_supplement_body = self._deconstruct_body(supplement_body[select_params.mime_type]) if supplement_body and select_params.mime_type in supplement_body else None
                    if deconstructed_supplement_body:
                        for prop in possible_body_properties:
                            if prop not in unconstructed_body:
                                unconstructed_body[prop] = deconstructed_supplement_body[prop] if prop in deconstructed_supplement_body else random_generator()()
                    body = {select_params.mime_type: self._construct_body(unconstructed_body, operation_id, select_params.mime_type)}
            else:
                parameters = None
                body = None

            # Assign header token if header agent is enabled and header is assigned
            header = {"Authorization": select_header} if select_header else None

            # Use body agent to select properties based on assigned parameters for body
            select_body_properties = {}
            if body:
                for mime, body_properties in body.items():
                    if type(body_properties) == dict:
                        select_properties = self.body_object_agent.get_action(operation_id, mime)
                        deconstructed_body = self._deconstruct_body(body_properties)
                        if select_properties:
                            new_bodies_properties = {prop: deconstructed_body[prop] for prop in deconstructed_body if prop in select_properties}
                            body[mime] = new_bodies_properties
                        else:
                            body[mime] = None
                        select_body_properties[mime] = select_properties

            # Mutate operation values
            mutate_operation = random.random() < self.mutation_rate
            mutated_parameter_names = False
            operation_props = self.operation_graph.operation_nodes[operation_id].operation_properties
            if mutate_operation:
                avail_primitives = len(self.successful_primitives.values())
                use_mutator = random.random() < 0.8 or (avail_primitives == 0 and operation_id not in complete_body_mappings)
                specific_method = None

                if use_mutator:
                    parameters, body, header, specific_method, mutated_parameter_names = self.mutate_values(
                        operation_props, parameters, body, header)
                else:
                    if random.random() < 0.2 and avail_primitives > 0:
                        parameters, body = self.assign_random_from_primitives(parameters, body, operation_id)
                    elif operation_id in complete_body_mappings:
                        body = self.assign_random_complete_body(body, operation_id, complete_body_mappings)
                    else:
                        parameters, body, header, specific_method, mutated_parameter_names = self.mutate_values(
                            operation_props, parameters, body, header)

                response = self.send_operation(operation_props, parameters, body, header, specific_method)
            else:
                response = self.send_operation(operation_props, parameters, body, header)

            # If invalid response do not process
            if response is None:
                continue

            # Only update table when using table values (so not mutated)
            if not mutate_operation:
                self.operation_agent.update_q_table(operation_id, self.determine_bad_response_reward(response))

                curr_Q_param, curr_Q_mime = self.parameter_agent.get_Q_curr(operation_id, select_params)
                next_Q_param, next_Q_mime = self.parameter_agent.get_Q_next(operation_id)
                curr_Q_body, next_Q_body = 0, 0
                if select_body_properties:
                    for mime, select_properties in select_body_properties.items():
                        curr_Q_body += self.body_object_agent.get_Q_curr(operation_id, mime, select_properties)
                        next_Q_body += self.body_object_agent.get_Q_next(operation_id, mime)

                curr_Q_data = self.data_source_agent.get_Q_curr(operation_id, data_source)
                next_Q_data = self.data_source_agent.get_Q_next(operation_id)

                curr_Q_header, next_Q_header = 0, 0
                if self.config.enable_header_agent:
                    curr_Q_header += self.header_agent.get_Q_curr(operation_id, select_header)
                    next_Q_header += self.header_agent.get_Q_next(operation_id)

                processed_value_action, used_dependent_params, used_dependent_body = None, None, None
                curr_Q_value_params, curr_Q_value_body, next_Q_value_params, next_Q_value_body, curr_Q_dependency_params, curr_Q_dependency_body, next_Q_dependency_params, next_Q_dependency_body = [], [], [], [], [], [], [], []
                if data_source == "DEPENDENCY" and (dependency_type == "EXPLORE" or dependency_type == "BEST"):
                    used_dependent_params = {}
                    if parameter_dependencies:
                        for parameter in parameters:
                            if parameter in select_params.req_params and parameter in parameter_dependencies:
                                used_dependent_params[parameter] = parameter_dependencies[parameter]
                    used_dependent_body = {}
                    if request_body_dependencies and unconstructed_body and select_params.mime_type in select_body_properties and \
                            select_body_properties[select_params.mime_type]:
                        for body_param in unconstructed_body:
                            if body_param in select_body_properties[
                                select_params.mime_type] and body_param in request_body_dependencies:
                                used_dependent_body[body_param] = request_body_dependencies[body_param]
                    curr_Q_dependency_params, curr_Q_dependency_body = self.dependency_agent.get_Q_curr(operation_id, used_dependent_params, used_dependent_body)
                    next_Q_dependency_params, next_Q_dependency_body = self.dependency_agent.get_Q_next(operation_id, used_dependent_params, used_dependent_body)
                elif data_source == "DEPENDENCY" and dependency_type == "RANDOM":
                    if not 200 <= response.status_code < 300:
                        continue
                    if parameter_dependencies:
                        for parameter, dependency_info in parameter_dependencies.items():
                            dependent_operation = dependency_info["dependent_operation"]
                            dependency_location = dependency_info["in_value"]
                            dependent_val = dependency_info["dependent_val"]
                            self.dependency_agent.add_new_dependency(operation_id, "params", parameter, dependent_operation, dependency_location, dependent_val)
                    if request_body_dependencies:
                        for body_param, dependency_info in request_body_dependencies.items():
                            dependent_operation = dependency_info["dependent_operation"]
                            dependency_location = dependency_info["in_value"]
                            dependent_val = dependency_info["dependent_val"]
                            self.dependency_agent.add_new_dependency(operation_id, "body", body_param, dependent_operation, dependency_location, dependent_val)

                # Calculate combined error through value decomposition
                Q_target = next_Q_data + next_Q_param + next_Q_mime + next_Q_body + next_Q_header + sum(next_Q_value_params) + sum(next_Q_value_body) + sum(next_Q_dependency_params) + sum(next_Q_dependency_body)
                Q_curr = curr_Q_data + curr_Q_param + curr_Q_mime + curr_Q_body + curr_Q_header + sum(curr_Q_value_params) + sum(curr_Q_value_body) + sum(curr_Q_dependency_params) + sum(curr_Q_dependency_body)
                td_error = self.determine_good_response_reward(response) + self.gamma * Q_target - Q_curr

                # Update Q-tables
                self.parameter_agent.update_Q_item(operation_id, select_params, td_error)
                if select_body_properties:
                    for mime, select_properties in select_body_properties.items():
                        self.body_object_agent.update_Q_item(operation_id, mime, select_properties, td_error)

                self.data_source_agent.update_Q_item(operation_id, data_source, td_error)

                if self.config.enable_header_agent:
                    self.header_agent.update_Q_item(operation_id, select_header, td_error)

                elif data_source == "DEPENDENCY" and (dependency_type == "EXPLORE" or dependency_type == "BEST"):
                    self.dependency_agent.update_Q_item(operation_id, used_dependent_params, used_dependent_body, td_error)

            # Update successful parameters to use for future operation dependencies
            if response is not None and response.ok and not mutated_parameter_names:
                print("Successful response!")
                if parameters and self.successful_parameters[operation_id]:
                    for param_name, param_val in parameters.items():
                        if param_name in self.successful_parameters[operation_id] and param_val not in self.successful_parameters[operation_id][param_name]:
                            self.successful_parameters[operation_id][param_name].append(param_val)
                if body and self.successful_bodies[operation_id]:
                    for mime, body_properties in body.items():
                        deconstructed_body = self._deconstruct_body(body_properties)
                        if deconstructed_body:
                            for prop_name, prop_val in deconstructed_body.items():
                                if prop_name in self.successful_bodies[operation_id] and prop_val not in self.successful_bodies[operation_id][prop_name]:
                                    self.successful_bodies[operation_id][prop_name].append(prop_val)
                if response.content and self.successful_responses[operation_id] is not None:
                    try:
                        response_content = json.loads(response.content)
                    except json.JSONDecodeError:
                        print("Error decoding JSON response content")
                        print("Response content: ", response.content)
                        response_content = None

                    deconstructed_response: Dict[str, List] = {}
                    self._deconstruct_response(response_content, deconstructed_response)

                    if deconstructed_response:
                        for response_prop, response_vals in deconstructed_response.items():
                            if response_prop in self.successful_responses[operation_id]:
                                for response_val in response_vals:
                                    if response_val not in self.successful_responses[operation_id][response_prop]:
                                        self.successful_responses[operation_id][response_prop].append(response_val)
                            else:
                                self.successful_responses[operation_id][response_prop] = response_vals
                                if self.dependency_agent.add_undocumented_responses(operation_id, response_prop) and "DEPENDENCY" not in self.data_source_agent.available_data_sources:
                                    self.data_source_agent.initialize_dependency_source()

                    else:
                        if operation_id not in self.successful_primitives:
                            self.successful_primitives[operation_id] = []
                        if isinstance(response_content, list):
                            for item in response_content:
                                if item not in self.successful_primitives[operation_id]:
                                    self.successful_primitives[operation_id].append(item)
                        elif response_content not in self.successful_primitives[operation_id]:
                            self.successful_primitives[operation_id].append(response_content)

            if response is not None:
                self.responses[response.status_code] += 1
                if operation_id not in self.operation_response_counter:
                    self.operation_response_counter[operation_id] = {response.status_code: 1}
                elif response.status_code not in self.operation_response_counter[operation_id]:
                    self.operation_response_counter[operation_id][response.status_code] = 1
                else:
                    self.operation_response_counter[operation_id][response.status_code] += 1

                if 500 <= response.status_code < 600:
                    if operation_id not in self.errors:
                        self.errors[operation_id] = 1
                    else:
                        self.errors[operation_id] += 1

                    data_signature = {
                        "parameters": parameters,
                        "body": body,
                        "operation_id": operation_id
                    }
                    if operation_id not in self.unique_errors:
                        self.unique_errors[operation_id] = [data_signature]
                    elif data_signature not in self.unique_errors[operation_id]:
                        self.unique_errors[operation_id].append(data_signature)

    def tui_output(self, start_time, operation_id):

        unique_processed_200s = set()
        for operation_idx, status_codes in self.operation_response_counter.items():
            for status_code in status_codes:
                if status_code // 100 == 2:
                    unique_processed_200s.add(operation_idx)
        not_hit_operations = set()
        for operation_idx in self.operation_graph.operation_nodes.keys():
            if operation_idx not in unique_processed_200s:
                not_hit_operations.add(operation_idx)

        unique_errors = 0
        for operation_idx in self.unique_errors:
            unique_errors += len(self.unique_errors[operation_idx])

        print("=========================================================================")
        print(f"Attempting operation: {operation_id}")
        print(f"Status Code Counter: {dict(self.responses)}")
        print(f"Number of unique server errors: {unique_errors}")
        print(f"Number of successful operations: {len(unique_processed_200s)}")
        print(f"Percentage of successful operations: {len(unique_processed_200s) / len(self.operation_graph.operation_nodes) * 100:.2f}%")
        print("Time remaining: ", max(round(self.time_duration - (time.time() - start_time), 3), 0.01))
        print("Percentage of time elapsed: ", str(round((time.time() - start_time) / self.time_duration * 100, 2)) + "%")

    def run(self):
        self.execute_operations()

def init_graph_ablation_1(spec_name: str, spec_path, embedding_model, config: Config | None = None) -> OperationGraph:
    spec_parser = SpecificationParser(spec_path=spec_path, spec_name=spec_name, config=config)
    api_url = get_api_url(spec_parser, local_test=True)
    operation_graph = OperationGraph(spec_path=spec_path, spec_name=spec_name, spec_parser=spec_parser, embedding_model=embedding_model)
    request_generator = RequestGenerator(operation_graph=operation_graph, api_url=api_url, is_naive=False)
    operation_graph.assign_request_generator(request_generator)
    return operation_graph

def generate_graph_ablation_1(spec_dir, spec_name, embedding_model, config: Config | None = None):
    print("Generating graph!")
    spec_path = f"{spec_dir}{spec_name}.yaml"
    operation_graph = init_graph_ablation_1(spec_name, spec_path, embedding_model, config=config)
    operation_graph.create_graph()
    print("Graph initialized!")
    return operation_graph

def perform_q_learning_ablation_1(operation_graph: OperationGraph, spec_name, duration):
    print("Initializing agents!")
    q_learning = Ablation1(
        operation_graph,
        alpha=operation_graph.config.q_learning.learning_rate,
        gamma=operation_graph.config.q_learning.discount_factor,
        epsilon=operation_graph.config.q_learning.max_exploration,
        time_duration=duration,
        mutation_rate=operation_graph.config.request_generation.mutation_rate,
    )
    q_learning.parameter_agent.initialize_q_table()
    q_learning.operation_agent.initialize_q_table()
    q_learning.body_object_agent.initialize_q_table()
    q_learning.data_source_agent.available_data_sources.remove("LLM")
    q_learning.data_source_agent.initialize_q_table()
    q_learning.dependency_agent.initialize_q_table()
    print("Starting Q-learning!")
    q_learning.run()
    print("Q-learning complete!")
    return q_learning

def execute_ablation_1(spec_dir, spec_name, duration, config: Config | None = None):
    """
    Perform ablation study 1: Remove LLM data source.
    Note: Ablation only works with yaml input files and with the header agent disabled in configurations.
    Runtime duration and specification location should be configured in the main method here.
    The remaining configurations (learning rate, etc...) are taken from the configurations.py file.
    This is meant as a benchmark.
    :return:
    """
    config = config if config is not None else get_config()
    embedding_model = EmbeddingModel()
    operation_graph = generate_graph_ablation_1(spec_dir, spec_name, embedding_model, config=config)
    q_learning = perform_q_learning_ablation_1(operation_graph, spec_name, duration)

if __name__ == "__main__":
    spec_dir = "../../aratrl-openapi/"
    spec_name = "project"
    duration = 1800
    execute_ablation_1(spec_dir, spec_name, duration)
