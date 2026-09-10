import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from prance import ResolvingParser, ValidationError as PranceValidationError
from openapi_spec_validator.validation.exceptions import (
    ValidationError as OASValidationError,  # type: ignore[attr-defined]
    OpenAPIValidationError,
)

from autoresttest.config import Config, get_config
from autoresttest.models import (
    OperationProperties,
    ParameterKey,
    ParameterProperties,
    ResponseProperties,
    SchemaProperties,
    to_dict_helper,
)


def json_spec_output(output_directory: Path, file_name: str, spec: Dict):
    """
    Create a testing JSON file from the specification parsing output.
    """
    output_file = output_directory / file_name
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(spec, file, ensure_ascii=False, indent=4)




def default_recursive_limit_handler(
    limit: int, parsed_url: Any, recursions: tuple[str, ...] = ()
) -> dict[str, Any]:
    """
    Prance invokes this handler when a circular/self-reference exceeds the recursion_limit.

    Parameters:
        limit: The configured recursion_limit value.
        parsed_url: The parsed URL of the reference that triggered the limit (contains the $ref path).
        recursions: Tuple of reference URLs that form the circular chain.

    Returns a placeholder schema to substitute for the circular reference.
    """
    # Extract the reference name from the parsed URL fragment (e.g., "#/components/schemas/Node" -> "Node")
    ref_name = "self-reference"
    if parsed_url and hasattr(parsed_url, "fragment") and parsed_url.fragment:
        ref_name = parsed_url.fragment.split("/")[-1]

    return {
        "type": "object",
        "title": f"Circular Reference: {ref_name}",
        "description": f"circular_self_reference:{ref_name}",
        "properties": {},
    }


class LenientResolvingParser(ResolvingParser):
    """
    ResolvingParser constructor calls:
    1. RefResolver.resolve_references()
    2. BaseParser._validate(self) -> openapi-spec-validator
    Override _validate to provide error on incorrect schemas instead of crashing
    """

    def _validate(self):
        try:
            super()._validate()
        except (
            PranceValidationError,
            OASValidationError,
            OpenAPIValidationError,
        ) as exc:
            logging.warning(
                "OpenAPI validation failed for %s; continuing with unvalidated "
                "specs. Error: %s",
                getattr(self, "url", None),
                exc,
            )
            self.valid = False


class SpecificationParser:
    """
    Class to parse a specification file and return a dictionary of all the operations and their properties.
    """

    def __init__(self, spec_path=None, spec_name=None, config: Config | None = None):
        self.config = config if config is not None else get_config()
        self.spec_path = spec_path
        self.spec_name = spec_name
        if spec_path is None:
            raise ValueError("No specification path provided.")

        self.resolving_parser = self._build_resolving_parser(spec_path)
        if self.resolving_parser.specification is None:
            raise ValueError("An error occurred during Prance specification parsing.")

        self.directory_path = (
            "specs/aratrl-openapi/"  # DEPRECATED - NOT USED IN EXECUTION
        )
        self.all_specs = {}

    def _build_resolving_parser(self, spec_path):
        parser_cls = (
            ResolvingParser if self.config.strict_validation else LenientResolvingParser
        )
        return parser_cls(
            spec_path,
            backend="openapi-spec-validator",  # AutoRestTest supports OAS 3.x
            strict=True,
            recursion_limit=self.config.recursion_limit,
            recursion_limit_handler=default_recursive_limit_handler,
        )

    def get_api_url(self) -> str:
        """
        Extract the server URL from the specification file.
        """
        spec = self.resolving_parser.specification
        if spec is None:
            raise ValueError("Specification not initialized successfully.")
        servers = spec.get("servers")
        if not servers:
            raise ValueError("No servers defined in the OpenAPI specification.")
        first_server = servers[0]
        if not isinstance(first_server, dict):
            raise ValueError("Invalid server definition in the OpenAPI specification.")
        url = first_server.get("url")
        if not url:
            raise ValueError("Server URL is missing in the OpenAPI specification.")
        return url

    def get_api_title(self) -> str | None:
        """
        Extract the title of the API from the specification file.
        """
        spec = self.resolving_parser.specification
        if spec is None:
            return None
        info = spec.get("info")
        if not isinstance(info, dict):
            return None
        title = info.get("title")
        return title if isinstance(title, str) else None

    def process_parameter_object_properties(
        self, properties: Dict | None
    ) -> Dict[str, SchemaProperties] | None:
        """
        Process the properties of a parameter of type object to return a dictionary of all the properties and their
        corresponding parameter values.
        """
        if properties is None or not isinstance(properties, dict):
            return None
        object_properties = {}
        for name, values in properties.items():
            object_properties.setdefault(
                name, self.process_parameter_schema(values)
            )  # check if this is correct, or if it should be process_parameter
        # Filter out None values from nested schema processing
        return {k: v for k, v in object_properties.items() if v is not None}

    def _infer_schema_type(self, schema: Dict) -> str | None:
        """Infer schema type from context when not explicitly provided."""
        if schema.get("type"):
            return schema.get("type")

        # Array indicators
        if schema.get("items") is not None:
            return "array"

        # Object indicators
        if schema.get("properties") is not None:
            return "object"
        if schema.get("additionalProperties") is not None:
            return "object"

        # Infer from enum values
        enum = schema.get("enum")
        if enum and len(enum) > 0:
            first_val = enum[0]
            if isinstance(first_val, bool):
                return "boolean"
            if isinstance(first_val, int):
                return "integer"
            if isinstance(first_val, float):
                return "number"
            if isinstance(first_val, str):
                return "string"

        # Numeric constraint indicators
        if any(
            schema.get(k) is not None
            for k in ("minimum", "maximum", "multipleOf", "exclusiveMinimum", "exclusiveMaximum")
        ):
            return "number"

        # String constraint indicators
        if any(schema.get(k) is not None for k in ("minLength", "maxLength", "pattern")):
            return "string"

        return None

    def process_parameter_schema(
        self, schema: Dict | None, description: str | None = None
    ) -> SchemaProperties | None:
        """
        Process the schema of a parameter to return a ValueProperties object
        """
        if not schema or not isinstance(schema, dict):
            return None

        value_properties = SchemaProperties(
            type=self._infer_schema_type(schema),
            format=schema.get("format"),
            description=schema.get("description") if not description else description,
            items=self.process_parameter_schema(
                schema.get("items")
            ),  # recursively process items
            properties=self.process_parameter_object_properties(
                schema.get("properties")
            ),
            required=schema.get("required") or [],
            default=schema.get("default"),
            enum=schema.get("enum") or [],
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            min_length=schema.get("minLength"),
            max_length=schema.get("maxLength"),
            pattern=schema.get("pattern"),
            max_items=schema.get("maxItems"),
            min_items=schema.get("minItems"),
            unique_items=schema.get("uniqueItems"),
            additional_properties=schema.get("additionalProperties"),
            nullable=schema.get("nullable"),
            read_only=schema.get("readOnly"),
            write_only=schema.get("writeOnly"),
            example=schema.get("example"),
            examples=schema.get("examples") or [],
        )
        return value_properties

    def process_parameter(self, parameter: Any) -> ParameterProperties:
        """
        Process an individual parameter to return a ParameterProperties object.
        """
        if not isinstance(parameter, dict):
            return ParameterProperties()
        parameter_properties = ParameterProperties(
            name=parameter.get("name") or "",
            in_value=parameter.get("in"),
            description=parameter.get("description"),
            required=parameter.get("required"),
            deprecated=parameter.get("deprecated"),
            allow_empty_value=parameter.get("allowEmptyValue"),
            style=parameter.get("style"),
            explode=parameter.get("explode"),
            allow_reserved=parameter.get("allowReserved"),
        )
        if parameter.get("schema"):
            parameter_properties.schema = self.process_parameter_schema(
                parameter.get("schema")
            )
        return parameter_properties

    def process_parameters(
        self, parameter_list
    ) -> Dict[ParameterKey, ParameterProperties]:
        """
        Process the parameters list to return a Dictionary with all its properties and values.
        """
        parameters = {}
        if parameter_list:
            for parameter in parameter_list:
                parameter_properties = self.process_parameter(parameter)
                if not parameter_properties.name:
                    continue
                key = (parameter_properties.name, parameter_properties.in_value)
                parameters[key] = parameter_properties
        return parameters

    def process_request_body(self, request_body) -> Dict[str, SchemaProperties]:
        """
        Process the request body to return a Dictionary with mime type and its properties and values.
        """

        request_body_properties = {}
        content = request_body.get("content")
        description = request_body.get("description")
        if content:
            for mime_type, mime_details in content.items():
                # if we need to check required list, do it here
                schema = mime_details.get("schema")
                if schema:
                    request_body_properties[mime_type] = self.process_parameter_schema(
                        schema, description
                    )

        return request_body_properties

    def process_responses(self, responses) -> Dict[str, ResponseProperties]:
        """
        Process the responses to return a Dictionary with status code and its properties and values.
        """
        response_properties = {}
        for status_code, response_details in responses.items():
            response_properties.setdefault(
                status_code,
                ResponseProperties(
                    status_code=status_code,
                    description=response_details.get("description"),
                ),
            )
            content = response_details.get("content")
            if content:
                for mime_type, mime_details in content.items():
                    # if we need to check required list, do it here
                    schema = mime_details.get("schema")
                    if schema:
                        response_properties[status_code].content[mime_type] = (
                            self.process_parameter_schema(schema)
                        )
        return response_properties

    def process_operation_details(
        self,
        http_method: str,
        endpoint_path: str,
        operation_details: Dict,
        operation_id: str,
        merged_parameters: List[Dict],
    ) -> OperationProperties:
        """
        Process the parameters and request body details within a given operation to return as OperationProperties object.
        """
        operation_properties = OperationProperties(
            operation_id=operation_id,
            endpoint_path=endpoint_path,
            http_method=http_method,
            summary=operation_details.get("summary"),
        )
        if merged_parameters:
            operation_properties.parameters = self.process_parameters(
                parameter_list=merged_parameters
            )

        if operation_details.get("requestBody"):
            operation_properties.request_body = self.process_request_body(
                request_body=operation_details.get("requestBody")
            )

        if operation_details.get("responses"):
            operation_properties.responses = self.process_responses(
                responses=operation_details.get("responses")
            )

        # maybe add security details?

        return operation_properties

    def _normalize_endpoint_path(self, endpoint_path: str) -> str:
        """
        Convert a path template to a safe string segment for fallback operationIds.
        """
        normalized = endpoint_path.replace("{", "").replace("}", "")
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_")
        return normalized or "root"

    def _resolve_operation_id(
        self,
        http_method: str,
        endpoint_path: str,
        operation_details: Dict,
        seen_ids: Set[str],
    ) -> str:
        """
        Use provided operationId when available; otherwise construct a unique, deterministic fallback.
        If a duplicate is detected, append an incrementing suffix to keep IDs unique.
        """
        provided_id = operation_details.get("operationId")
        candidate = provided_id if provided_id else None

        if not candidate:
            normalized_path = self._normalize_endpoint_path(endpoint_path)
            candidate = f"{http_method.lower()}_{normalized_path}"

        base_candidate = candidate
        suffix = 1
        while candidate in seen_ids:
            candidate = f"{base_candidate}_{suffix}"
            suffix += 1

        if provided_id and candidate != provided_id:
            logging.warning(
                "Duplicate operationId '%s' for %s %s; using '%s' to avoid collision.",
                provided_id,
                http_method,
                endpoint_path,
                candidate,
            )

        return candidate

    def _merge_parameters(
        self, path_parameters: List[Dict], operation_parameters: List[Dict]
    ) -> List[Dict]:
        """
        Merge path-level and operation-level parameters, letting operation-level definitions override
        matching path-level parameters (by (name, in)).
        """
        merged: Dict[ParameterKey, Dict] = {}
        for raw_param in path_parameters or []:
            if isinstance(raw_param, dict):
                name = raw_param.get("name")
                if name is not None:
                    merged[(name, raw_param.get("in"))] = raw_param
        for raw_param in operation_parameters or []:
            if isinstance(raw_param, dict):
                name = raw_param.get("name")
                if name is not None:
                    merged[(name, raw_param.get("in"))] = raw_param
        return list(merged.values())

    def parse_specification(self) -> Dict[str, OperationProperties]:
        """
        Parse the specification file to return a dictionary of all the operations and their properties.

        The key of the dictionary is the operationId and the value is an OperationProperties object.
        """
        supported_methods = {"get", "post", "put", "delete", "head", "options", "patch"}
        operation_collection = {}
        seen_ids: Set[str] = set()
        spec = self.resolving_parser.specification
        if spec is None:
            return operation_collection
        spec_paths = spec.get("paths", {})
        if not isinstance(spec_paths, dict):
            spec_paths = {}
        for endpoint_path, endpoint_details in spec_paths.items():
            if not isinstance(endpoint_details, dict):
                continue
            path_level_parameters = endpoint_details.get("parameters", [])
            for http_method, operation_details in endpoint_details.items():
                if http_method in supported_methods:
                    merged_parameters = self._merge_parameters(
                        path_level_parameters, operation_details.get("parameters", [])
                    )
                    operation_id = self._resolve_operation_id(
                        http_method, endpoint_path, operation_details, seen_ids
                    )
                    seen_ids.add(operation_id)
                    operation_properties = self.process_operation_details(
                        http_method,
                        endpoint_path,
                        operation_details,
                        operation_id,
                        merged_parameters,
                    )
                    operation_collection[operation_id] = operation_properties
        return operation_collection

    def parse_all_specifications(self) -> dict[str, dict[str, OperationProperties]]:
        """
        Parse all the specification files in the directory to return a dictionary of all the operations and their properties.
        """
        for file_name in os.listdir(self.directory_path):
            print("Specification parsing for file: ", file_name)
            self.spec_path = os.path.join(self.directory_path, file_name)
            self.resolving_parser = self._build_resolving_parser(self.spec_path)
            self.all_specs[file_name] = self.parse_specification()
            # print("Output: " + str(output))

        return self.all_specs

    def all_json_spec_output(self):
        """
        Create a testing JSON file from the specification parsing output.
        """
        all_specs = {
            file_name: to_dict_helper(spec)
            for file_name, spec in self.all_specs.items()
        }

        output_directory = Path("./testing_output")
        output_directory.mkdir(parents=True, exist_ok=True)

        for file_name, spec in all_specs.items():
            json_spec_output(output_directory, file_name, spec)

    def single_json_spec_output(self):
        output_directory = Path("./testing_output")
        output_directory.mkdir(parents=True, exist_ok=True)
        output = self.parse_specification()
        file_name = self.spec_name if self.spec_name else "spec_output.json"
        json_spec_output(output_directory, file_name, to_dict_helper(output))


if __name__ == "__main__":
    # testing
    # spec_path = os.path.normpath("../../aratrl-openapi/project.yaml")
    spec_path = os.path.normpath("kafka-rest.yaml")
    spec_parser = SpecificationParser(spec_name="project", spec_path=spec_path)
    spec_parser.parse_specification()
    print(spec_parser.parse_specification())
    # spec_parser.single_json_spec_output()
    # spec_parser.parse_all_specifications()
    # spec_parser.all_json_spec_output()

    # spec_parser = SpecificationParser(spec_path="../specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    # spec_parser.single_json_spec_output()

    # output = spec_parser.parse_specification()
    # print(output["fetchPostTranslationalModificationsByPtmFilterPOST"])
    # print(output["endpoint-add-tracks-to-playlist"])
    # print(output["endpoint-get-playlist"])
    # print(output["endpoint-remove-tracks-playlist"])
