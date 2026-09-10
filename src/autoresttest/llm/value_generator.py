import json
import random
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from autoresttest.config import Config, get_config
from autoresttest.models import (
    OperationProperties,
    ParameterKey,
    ParameterProperties,
    RequestData,
    RequestRequirements,
    RequestResponse,
    SchemaProperties,
)
from autoresttest.prompts import (
    ENUM_EXAMPLE_CONSTRAINT_PROMPT,
    FAILED_PARAMETER_MATCHINGS_PROMPT,
    FAILED_PARAMETER_RESPONSE_PROMPT,
    FEWSHOT_PARAMETER_GEN_PROMPT,
    FEWSHOT_REQUEST_BODY_GEN_PROMPT,
    IDENTIFY_AUTHENTICATION_GEN_PROMPT,
    IDENTIFY_AUTHENTICATION_SYSTEM_MESSAGE,
    PARAMETER_NECESSITY_PROMPT,
    PARAMETER_REQUIREMENTS_PROMPT,
    PARAMETERS_GEN_PROMPT,
    PARAMETERS_GEN_SYSTEM_MESSAGE,
    REQUEST_BODY_GEN_PROMPT,
    REQUEST_BODY_GEN_SYSTEM_MESSAGE,
    RETRY_PARAMETER_REQUIREMENTS_PROMPT,
    VALUE_AGENT_BODY_FEWSHOT_PROMPT,
    VALUE_AGENT_PARAMS_FEWSHOT_PROMPT,
    get_informed_agent_body_prompt,
    get_informed_agent_params_prompt,
    get_value_agent_body_prompt,
    get_value_agent_params_prompt,
    template_gen_prompt,
)
from autoresttest.utils import (
    attempt_fix_json,
    param_key_to_label,
    remove_nulls,
)

from .llm import LanguageModel



def randomize_boolean():
    return random.choice([True, False])


def randomize_null():
    return None


def randomize_integer():
    percent = random.randint(1, 100)
    if percent <= 60:
        return random.randint(0, 20)
    elif percent <= 90:
        return random.randint(0, 1000)
    else:
        return random.randint(-(2**10), (2**10))


def randomize_float():
    percent = random.randint(1, 100)
    if percent <= 60:
        return random.uniform(0, 20)
    elif percent <= 90:
        return random.uniform(0, 1000)
    else:
        return random.uniform(-(2**10), (2**10))


def randomize_string():
    percent = random.randint(1, 100)
    if percent <= 60:
        length = random.randint(1, 8)
    elif percent <= 90:
        length = random.randint(4, 20)
    else:
        length = random.randint(1, 50)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def randomize_array():
    percent = random.randint(1, 100)
    if percent <= 60:
        length = random.randint(1, 8)
    elif percent <= 90:
        length = random.randint(4, 20)
    else:
        length = random.randint(0, 50)
    return [random.randint(-9999, 9999) for _ in range(length)]


def randomize_object():
    if random.randint(1, 100) <= 90:
        length = random.randint(4, 10)
    else:
        length = random.randint(0, 50)
    return {
        random.choice(string.ascii_letters): random.randint(-9999, 9999)
        for _ in range(length)
    }


def randomized_array_length():
    if random.randint(0, 100) <= 90:
        return random.randint(4, 10)
    else:
        return random.randint(0, 50)


def identify_generator(value: Any):
    generators = {
        "integer": randomize_integer,
        "float": randomize_float,
        "number": randomize_float,
        "boolean": randomize_boolean,
        "string": randomize_string,
        "array": randomize_array,
        "object": randomize_object,
        "null": randomize_null,
    }
    return generators.get(value) or random_generator()


def random_generator():
    generators = {
        "integer": randomize_integer,
        "float": randomize_float,
        "number": randomize_float,
        "boolean": randomize_boolean,
        "string": randomize_string,
        "array": randomize_array,
        "object": randomize_object,
        "null": randomize_null,
    }
    return random.choice(list(generators.values()))


class NaiveValueGenerator:
    def __init__(
        self,
        parameters: Dict[ParameterKey, ParameterProperties],
        request_body: Dict[str, SchemaProperties] | None,
    ):
        self.parameters: Dict[ParameterKey, ParameterProperties] = parameters
        self.request_body: Dict[str, SchemaProperties] | None = request_body

    def generate_value(self, item_properties: SchemaProperties) -> Any:
        if item_properties is None:
            return None

        item_type = getattr(item_properties, "type", None)
        props = getattr(item_properties, "properties", None)
        if props is not None and not isinstance(props, dict):
            props = None
        if item_type == "object" or props is not None:
            return {
                item_name: self.generate_value(prop_schema)
                for item_name, prop_schema in (props or {}).items()
            }
        items = getattr(item_properties, "items", None)
        if not isinstance(items, SchemaProperties):
            items = None
        if item_type == "array":
            if items is not None:
                return [
                    self.generate_value(items) for _ in range(randomized_array_length())
                ]
            else:
                return randomize_array()
        elif items is not None:
            return [
                self.generate_value(items) for _ in range(randomized_array_length())
            ]
        generator = identify_generator(item_type) if item_type else random_generator()
        return generator()

    def generate_parameters(self) -> Dict[ParameterKey, Any]:
        query_parameters = {}
        for parameter_name, parameter_properties in self.parameters.items():
            schema = parameter_properties.schema
            if schema is not None:
                randomized_value = self.generate_value(schema)
            else:
                randomized_value = random_generator()()
            query_parameters[parameter_name] = randomized_value
        return query_parameters

    def generate_request_body(self):
        if not self.request_body:
            return None
        request_properties = {}
        for item_name, item_properties in self.request_body.items():
            randomized_value = self.generate_value(item_properties)
            request_properties[item_name] = randomized_value  # save diff mime types
        return request_properties


@dataclass
class PromptData:
    GEN_PROMPT: str
    FEWSHOT_PROMPT: str
    schema: Dict
    select_params: Dict = field(default_factory=dict)
    is_request_body: bool = False
    response: requests.Response | None = None
    failed_mappings: Dict = field(default_factory=dict)


class SmartValueGenerator:
    def __init__(
        self,
        operation_properties: OperationProperties,
        requirements: Optional[RequestRequirements] = None,
        engine=None,
        temperature=None,
        config: Config | None = None,
    ):
        self.config = config if config is not None else get_config()
        self.operation_properties: OperationProperties = operation_properties
        self.processed_operation = remove_nulls(operation_properties.to_dict())
        self.parameters_raw: Dict[ParameterKey, ParameterProperties] = (
            operation_properties.parameters or {}
        )
        self.parameter_lookup: Dict[str, ParameterKey] = {
            param_key_to_label(key): key for key in self.parameters_raw.keys()
        }
        # Fallback lookup: plain parameter name -> ParameterKey (for when LLM strips ::location suffix)
        self.parameter_name_lookup: Dict[str, ParameterKey] = {
            key[0]: key for key in self.parameters_raw.keys()
        }
        self.parameters: Dict[str, Dict] = {
            label: remove_nulls(param.to_dict())
            for label, param in (
                (param_key_to_label(key), param)
                for key, param in self.parameters_raw.items()
            )
        }
        self.request_body: Dict[str, Dict] | None = self.processed_operation.get(
            "request_body"
        )
        self.summary: str = self.processed_operation.get("summary")
        self.language_model = LanguageModel(
            engine=engine, temperature=temperature, config=self.config
        )
        self.parameter_requirements_raw: Dict[ParameterKey, Any] = (
            requirements.parameter_requirements if requirements else {}
        )
        self.parameter_requirements_labels: Dict[str, Any] = {
            param_key_to_label(key): value
            for key, value in self.parameter_requirements_raw.items()
        }
        self.request_body_reqs: Dict[str, Any] = (
            requirements.request_body_requirements if requirements else {}
        )
        self.parameters_reqs: Dict[str, Any] = self.parameter_requirements_labels

    def _format_param_dict_for_prompt(self, params: Optional[Dict]) -> Dict:
        """
        Convert a parameter dict to a format suitable for LLM prompts.

        Accepts dicts with either:
        - ParameterKey tuples as keys (converted via param_key_to_label)
        - String keys (used as-is, e.g., for body property names)

        This dual-type support is intentional since the codebase uses ParameterKey
        for operation parameters but strings for request body properties.
        """
        if not params:
            return {}
        formatted = {}
        for key, value in params.items():
            if isinstance(key, tuple):
                label = param_key_to_label(key)
            else:
                label = str(key)
            formatted[label] = value
        return formatted

    def _compose_parameter_gen_prompt(self, prompt_data: PromptData, necessary=False):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        FEWSHOT_PROMPT = prompt_data.FEWSHOT_PROMPT
        schema = prompt_data.schema
        select_params = prompt_data.select_params
        is_request_body = prompt_data.is_request_body

        prompt = f"{GEN_PROMPT}\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)

        if necessary:
            prompt += (
                PARAMETER_NECESSITY_PROMPT + "\n".join(select_params.keys()) + "\n\n"
            )
        else:
            prompt += (
                PARAMETER_REQUIREMENTS_PROMPT + "\n".join(select_params.keys()) + "\n\n"
            )

        prompt += "Reminder:\n" + ENUM_EXAMPLE_CONSTRAINT_PROMPT + "\n"

        if FEWSHOT_PROMPT:
            prompt += "Here are some examples of creating values from specifications:\n"
            prompt += FEWSHOT_PROMPT + "\n"

        if is_request_body:
            prompt += "REQUEST_BODY VALUES:\n"
        else:
            prompt += "PARAMETER VALUES:\n"

        return prompt

    def _compose_retry_parameter_gen_prompt(self, prompt_data: PromptData):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        FEWSHOT_PROMPT = prompt_data.FEWSHOT_PROMPT
        schema = prompt_data.schema
        select_params = prompt_data.select_params
        is_request_body = prompt_data.is_request_body
        response = prompt_data.response
        failed_mappings = prompt_data.failed_mappings

        if not is_request_body:
            failed_mappings = self._format_param_dict_for_prompt(failed_mappings)

        prompt = f"{GEN_PROMPT}\n{FEWSHOT_PROMPT}\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)
        prompt += (
            RETRY_PARAMETER_REQUIREMENTS_PROMPT
            + "\n".join(select_params.keys())
            + "\n\n"
        )
        prompt += (
            FAILED_PARAMETER_MATCHINGS_PROMPT
            + json.dumps(failed_mappings, indent=2)
            + "\n"
        )
        if response is not None:
            prompt += FAILED_PARAMETER_RESPONSE_PROMPT + response.text + "\n\n"
        if is_request_body:
            prompt += "REQUEST_BODY VALUES:\n"
        else:
            prompt += "PARAMETERS VALUES:\n"
        # print("Prompt: ", prompt)
        return prompt

    def compose_informed_value_prompt(
        self, prompt_data: PromptData, responses: List[RequestResponse]
    ):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        schema = prompt_data.schema
        is_request_body = prompt_data.is_request_body
        few_shot_prompt = prompt_data.FEWSHOT_PROMPT

        prompt = f"{GEN_PROMPT}\n\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)
        if is_request_body:
            prompt += get_informed_agent_body_prompt() + "\n"
            for request_response in responses:
                if request_response is not None:
                    if request_response.request.request_body:
                        prompt += f"PAST REQUEST BODY: {request_response.request.request_body}\n"
                    prompt += f"STATUS CODE: {request_response.response.status_code}\n"
                    prompt += f"RESPONSE: {request_response.response.text[:1000]}\n\n"

        else:
            prompt += get_informed_agent_params_prompt() + "\n"
            for request_response in responses:
                if request_response is not None:
                    formatted_params = self._format_param_dict_for_prompt(
                        request_response.request.parameters
                    )
                    prompt += f"PAST PARAMETERS: {formatted_params}\n"
                    prompt += f"STATUS CODE: {request_response.response.status_code}\n"
                    prompt += f"RESPONSE: {request_response.response.text[:1000]}\n\n"

        prompt += "Regardless of the past responses:"
        prompt += ENUM_EXAMPLE_CONSTRAINT_PROMPT + "\n"

        prompt += "Here are some examples of creating values from specifications:\n"
        prompt += few_shot_prompt + "\n"

        if is_request_body:
            prompt += "REQUEST_BODY VALUES:\n"
        else:
            prompt += "PARAMETER VALUES:\n"
        return prompt

    def _compose_auth_gen_prompt(self, schema):
        prompt = IDENTIFY_AUTHENTICATION_GEN_PROMPT
        prompt += template_gen_prompt(summary=self.summary, schema=schema) + "\n"
        prompt += "AUTHENTICATION PARAMETERS:\n"
        # print("Prompt: ", prompt)
        return prompt

    def _isolate_nonreq_params(self, schema: Dict[str, Dict], is_request_body=False):
        if not isinstance(schema, dict):
            return {}
        nonreq_params = {}
        for param_name, param_properties in schema.items():
            if not is_request_body and param_name not in self.parameters_reqs:
                nonreq_params[param_name] = param_properties
            if is_request_body and param_name not in self.request_body_reqs:
                nonreq_params[param_name] = param_properties
        return nonreq_params

    def _isolate_nonreq_request_body(self, schema: Dict) -> Dict:
        properties = schema.get("properties")
        items = schema.get("items")
        if properties:
            # NOTE: We do not handle nested objects
            nonreq_request_body = self._isolate_nonreq_params(properties)
        elif items:
            nonreq_request_body = self._isolate_nonreq_request_body(items)
        else:
            nonreq_request_body = self._isolate_nonreq_params(schema)
        return nonreq_request_body

    def _form_parameter_gen_prompt(
        self, schema: Dict, is_request_body: bool, necessary: bool = False
    ):
        if is_request_body:
            prompt_data = PromptData(
                GEN_PROMPT=REQUEST_BODY_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=necessary)
        else:
            prompt_data = PromptData(
                GEN_PROMPT=PARAMETERS_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_PARAMETER_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_params(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=necessary)

    def _form_retry_parameter_gen_prompt(
        self,
        schema: Dict,
        failed_mappings: Dict,
        response: requests.Response,
        is_request_body: bool,
    ):
        if is_request_body:
            prompt_data = PromptData(
                GEN_PROMPT=REQUEST_BODY_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=is_request_body,
                response=response,
                failed_mappings=failed_mappings,
            )
            return self._compose_retry_parameter_gen_prompt(prompt_data)
        else:
            prompt_data = PromptData(
                GEN_PROMPT=PARAMETERS_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_PARAMETER_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_params(schema),
                is_request_body=is_request_body,
                response=response,
                failed_mappings=failed_mappings,
            )
            return self._compose_retry_parameter_gen_prompt(prompt_data)

    def _form_value_agent_prompt(
        self, schema: Dict, is_request_body: bool, num_values: int
    ):
        if is_request_body:
            prompt_data = PromptData(
                GEN_PROMPT=get_value_agent_body_prompt(num_values),
                FEWSHOT_PROMPT=VALUE_AGENT_BODY_FEWSHOT_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=False)
        else:
            prompt_data = PromptData(
                GEN_PROMPT=get_value_agent_params_prompt(num_values),
                FEWSHOT_PROMPT=VALUE_AGENT_PARAMS_FEWSHOT_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_params(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=True)

    def _validate_parameters(self, schema: Optional[Dict]) -> Dict[ParameterKey, Any]:
        if schema is None:
            return {}
        parameters: Dict[ParameterKey, Any] = {}
        for parameter_name, parameter_value in schema.items():
            param_key = self.parameter_lookup.get(parameter_name)
            if param_key and param_key not in self.parameter_requirements_raw:
                parameters[param_key] = parameter_value
        parameters.update(self.parameter_requirements_raw)
        return parameters

    def generate_parameters(self, necessary=False) -> Optional[Dict[ParameterKey, Any]]:
        """
        Uses the OpenAI language model to generate values for the parameters using JSON outputs
        :return: A dictionary of the generated parameters
        """
        if self.parameters is None or len(self.parameters) == 0:
            return None

        parameter_prompt = self._form_parameter_gen_prompt(
            schema=self.parameters, is_request_body=False, necessary=necessary
        )
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            if not generated_parameters or not generated_parameters.strip():
                generated_parameters = {}
            else:
                generated_parameters = attempt_fix_json(
                    generated_parameters, config=self.config
                )
        parameter_matchings = self._validate_parameters(
            generated_parameters.get("parameters") if isinstance(generated_parameters, dict) else None
        )
        return parameter_matchings

    def validate_request_body(self, schema: Any) -> Any:
        if schema is None:
            return {}
        if type(schema) is dict:
            # NOTE: We do not handle nested objects
            schema.update(self.request_body_reqs)
            return schema
        elif type(schema) is list:
            for i in range(len(schema)):
                schema[i] = self.validate_request_body(schema[i])
        return schema

    def generate_request_body(self, necessary=False) -> Optional[Dict[str, Any]]:
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_parameter_gen_prompt(
                schema=schema, is_request_body=True, necessary=necessary
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                if not generated_request_body or not generated_request_body.strip():
                    generated_request_body = {}
                else:
                    generated_request_body = attempt_fix_json(
                        generated_request_body, config=self.config
                    )
            validated_request_body = self.validate_request_body(
                generated_request_body.get("request_body") if isinstance(generated_request_body, dict) else None
            )
            if validated_request_body:  # Only add if we got valid content
                request_body[mime_type] = validated_request_body
        return request_body  # Returns {} if all mime types failed

    def generate_retry_parameters(
        self, failed_request_data: RequestData, response: requests.Response
    ) -> Optional[Dict[ParameterKey, Any]]:
        """
        Uses the OpenAI language model to generate values for the parameters using JSON outputs
        :return: A dictionary of the generated parameters
        """
        if self.parameters is None or len(self.parameters) == 0:
            return None

        parameter_prompt = self._form_retry_parameter_gen_prompt(
            schema=self.parameters,
            failed_mappings=failed_request_data.parameters or {},
            response=response,
            is_request_body=False,
        )
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            if not generated_parameters or not generated_parameters.strip():
                generated_parameters = {}
            else:
                generated_parameters = attempt_fix_json(
                    generated_parameters, config=self.config
                )
        parameter_matchings = self._validate_parameters(
            generated_parameters.get("parameters") if isinstance(generated_parameters, dict) else None
        )
        return parameter_matchings

    def generate_retry_request_body(
        self, failed_request_data: RequestData, response: requests.Response
    ) -> Optional[Dict[str, Any]]:
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_retry_parameter_gen_prompt(
                schema=schema,
                failed_mappings=(
                    failed_request_data.request_body.get(mime_type, {})
                    if failed_request_data.request_body
                    else {}
                ),
                response=response,
                is_request_body=True,
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                if not generated_request_body or not generated_request_body.strip():
                    generated_request_body = {}
                else:
                    generated_request_body = attempt_fix_json(
                        generated_request_body, config=self.config
                    )
            validated_request_body = self.validate_request_body(
                generated_request_body.get("request_body") if isinstance(generated_request_body, dict) else None
            )
            if validated_request_body:  # Only add if we got valid content
                request_body[mime_type] = validated_request_body
        return request_body  # Returns {} if all mime types failed

    def determine_auth_params(self):
        """
        Determines if the operation consists of any authentication information sent as parameters in either the query or the request body
        :return:
        """
        auth_prompt = self._compose_auth_gen_prompt(self.processed_operation)
        auth_parameters = self.language_model.query(
            user_message=auth_prompt,
            system_message=IDENTIFY_AUTHENTICATION_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            auth_parameters = json.loads(auth_parameters)
        except json.JSONDecodeError:
            if not auth_parameters or not auth_parameters.strip():
                auth_parameters = {}
            else:
                auth_parameters = attempt_fix_json(
                    auth_parameters, config=self.config
                )
        return auth_parameters.get("authentication_parameters") if isinstance(auth_parameters, dict) else None

    def _validate_value_params(
        self, schema: Optional[Dict]
    ) -> Dict[ParameterKey, List[Any]]:
        if schema is None:
            return {}
        param_mappings: Dict[ParameterKey, List[Any]] = defaultdict(list)
        for param_name, param_values in schema.items():
            # Try exact match first (e.g., "name::query"), then fallback to plain name (e.g., "name")
            param_key = self.parameter_lookup.get(param_name)
            if param_key is None:
                # Fallback: LLM may have stripped the ::location suffix
                param_key = self.parameter_name_lookup.get(param_name)
            if param_key in self.parameters_raw:
                for param_value in param_values.values():
                    param_mappings[param_key].append(param_value)
        return param_mappings

    def generate_value_agent_params(
        self, num_values: int
    ) -> Dict[ParameterKey, List[Any]]:
        """

        :param num_values:
        :return: A LIST of parameter mappings (dicts) for the operation; should have num_values items in list where each list has the parameter mappings
        """
        if self.parameters is None or len(self.parameters) == 0:
            return {}

        parameter_prompt = self._form_value_agent_prompt(
            schema=self.parameters, is_request_body=False, num_values=num_values
        )
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            if not generated_parameters or not generated_parameters.strip():
                generated_parameters = {}
            else:
                generated_parameters = attempt_fix_json(
                    generated_parameters, config=self.config
                )
        parameter_matchings = self._validate_value_params(
            generated_parameters.get("parameters") if isinstance(generated_parameters, dict) else None
        )
        return parameter_matchings

    def _validate_value_body(self, schema: Optional[Dict]) -> List:
        if schema is None:
            return []
        values = [body for body in schema.values()]
        return values

    def generate_value_agent_body(self, num_values: int) -> Dict[str, List]:
        """

        :param num_values:
        :return: A LIST of request body mappings (dicts) for the operation; should have num_values items in list where each list has the request body mappings
        """
        if self.request_body is None or len(self.request_body) == 0:
            return {}

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_value_agent_prompt(
                schema=schema, is_request_body=True, num_values=num_values
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                if not generated_request_body or not generated_request_body.strip():
                    generated_request_body = {}
                else:
                    generated_request_body = attempt_fix_json(
                        generated_request_body, config=self.config
                    )
            validated_request_body = self._validate_value_body(
                generated_request_body.get("request_body") if isinstance(generated_request_body, dict) else None
            )
            request_body[mime_type] = validated_request_body
        return request_body

    def generate_informed_value_agent_body(
        self, num_values: int, responses: List[RequestResponse]
    ) -> dict[str, Any]:
        if self.request_body is None or len(self.request_body) == 0:
            return {}

        request_body = {}
        for mime_type, schema in self.request_body.items():
            prompt_data = PromptData(
                GEN_PROMPT=get_value_agent_body_prompt(num_values),
                FEWSHOT_PROMPT=VALUE_AGENT_BODY_FEWSHOT_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=True,
            )
            request_body_prompt = self.compose_informed_value_prompt(
                prompt_data, responses
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                if not generated_request_body or not generated_request_body.strip():
                    generated_request_body = {}
                else:
                    print("Handling a JSON decode error...")
                    generated_request_body = attempt_fix_json(
                        generated_request_body, config=self.config
                    )
            validated_request_body = self._validate_value_body(
                generated_request_body.get("request_body") if isinstance(generated_request_body, dict) else None
            )
            request_body[mime_type] = validated_request_body
        return request_body

    def generate_informed_value_agent_params(
        self, num_values: int, responses: List[RequestResponse]
    ) -> Dict[ParameterKey, List[Any]]:
        if self.parameters is None or len(self.parameters) == 0:
            return {}

        prompt_data = PromptData(
            GEN_PROMPT=get_value_agent_params_prompt(num_values),
            FEWSHOT_PROMPT=VALUE_AGENT_PARAMS_FEWSHOT_PROMPT,
            schema=self.parameters,
            select_params=self._isolate_nonreq_params(self.parameters),
            is_request_body=False,
        )
        parameter_prompt = self.compose_informed_value_prompt(prompt_data, responses)
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            if not generated_parameters or not generated_parameters.strip():
                generated_parameters = {}
            else:
                generated_parameters = attempt_fix_json(
                    generated_parameters, config=self.config
                )
        parameter_matchings = self._validate_value_params(
            generated_parameters.get("parameters") if isinstance(generated_parameters, dict) else None
        )
        return parameter_matchings
