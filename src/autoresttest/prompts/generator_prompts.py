import json
from typing import Dict

REQUEST_BODY_GEN_PROMPT = """
Given a summary of an operation its request body schema from its OpenAPI Specification, generate a valid context-aware request body for the operation. Return the answer as a JSON object with the following structure:
{
    "request_body": [correct request body]
}
In the case where the request body is an object, return a correctly formatted object as the [correct request body] value. In the object, include all required fields as specified from the OpenAPI Specification for object request bodies, but otherwise include/exclude optional properties to ensure the object is accepted.
In the case where the request body is an array, use a list as the request_body value.
Do not solely rely on the given constraint values, and ensure you read the associated descriptions for maximum accuracy. Use any provided example values to guide your generation formatting."""

PARAMETERS_GEN_PROMPT = """
Given a summary of an operation and its parameters schema from its OpenAPI Specification, generate valid context-aware values for the parameters of the operation. Return the answer as a JSON object with the following structure:
{
    "parameters": {
        "[parameter1]": [value1],
        ...
        "[parameterN]": [valueN]
    }
}
IMPORTANT: Use the parameter keys EXACTLY as they appear in the SPECIFICATION object (the keys with "::" format combining name and location, e.g., "name::query", "productId::path"). Do NOT use plain parameter names without the location suffix.
In the case where a given parameter is an object, use an object with keys to represent the object field names and values to represent their respective field values as the parameter value. 
In the case where a given parameter is an array, use a list as the parameter value. Do not generate lists with more than two items.
Do not solely rely on the given constraint values, and ensure you read the associated descriptions for maximum accuracy. Use any provided example values to guide your generation formatting. 
It is vital that you generate values for any parameter that is indicated as required in the OpenAPI Specification."""

VALUE_AGENT_BODY_PROMPT = """
Given a summary of an operation its request body schema from its OpenAPI Specification, generate [insert number] different valid context-aware request bodies for the operation. Return the answer as a JSON object with the following structure:
{
    "request_body": {
        "request_body1": [correct request body 1],
        ...
        "request_body10": [correct request body 10]
    }
}
In the case where the request body is an object, return a correctly formatted object as the [correct request body] value. In the object, you must include every possible object property.
In the case where the request body is an array, use a list as the request body value, with the correct values for each item in the array. Do not generate lists with more than two items.
If the OpenAPI Specification includes examples in an example field or description, you must include some of the example values in your generated values. If the OpenAPI Specification includes enums, ensure that your generated values are within the enum values.
Do not solely rely on the given constraint values, and ensure you read the associated descriptions for maximum accuracy."""

VALUE_AGENT_PARAMETERS_PROMPT = """
Given a summary of an operation and its parameters schema from its OpenAPI Specification, generate [insert number] different valid context-aware values for the parameters of the operation. Return the answer as a JSON object with the following structure:
{
    "parameters": {
        "[parameter1]": {
            "value1": [value1],
            ...
            "value10": [value10]
        },
        "[parameter2]": {
            "value1": [value1],
            ...
            "value10": [value10]
        },
        ...
        "[parameterN]": {
            "value1": [value1],
            ...
            "value10": [value10]
        }
    }
}
IMPORTANT: Use the parameter keys EXACTLY as they appear in the SPECIFICATION object (the keys with "::" format combining name and location, e.g., "name::query", "productId::path"). Do NOT use plain parameter names without the location suffix.
In the case where a given parameter is an object, use an object with keys to represent the object field names and values to represent their respective field values as the parameter value.
In the case where a given parameter is an array, use a list as the parameter value. Do not generate lists with more than two items.
If the OpenAPI Specification includes examples in an example field or description, you must include some of the example values in your generated values. If the OpenAPI Specification includes enums, ensure that your generated values are within the enum values.
Do not solely rely on the given constraint values, and ensure you read the associated descriptions for maximum accuracy."""

INFORMED_VALUE_AGENT_PROMPT = """
To better inform your generation, here is a list of [replace_type] that you generated values for, along with their server responses. Use whether the past values generated an error to ensure that your new generated values are compatible with the operation.
Status codes that indicate a successful operation are of the form 2XX. Status codes that indicate a failed operation are of the form 4XX or 5XX. Use the responses to determine if the values you generated were correct or incorrect. You only need to consider server responses that indicate problems with parameter values, and not anything else.
"""

VALUE_AGENT_PARAMS_FEWSHOT_PROMPT = """
EXAMPLE 1
SPECIFICATION:
{
    "personId::path": {
        "name": "personId",
        "in_value": "path",
        "description": "The ID of the person. Must be in the form of their name and number. For example John123.",
        "schema": {
            "type": "string"
        },
        "required": true
    }
}
PARAMETER VALUES:
{
    "parameters": {
        "personId::path": {
            "value1": "John123",
            "value2": "Jane456",
            "value3": "Jack789",
            "value4": "Bob101",
            "value5": "Alice202",
            "value6": "Sam777",
            "value7": "Lily001",
            "value8": "Tom999",
            "value9": "Sara111",
            "value10": "Mike123"
        }
    }
}

EXAMPLE 2
SPECIFICATION:
{
    "language::query": {
        "name": "language",
        "in_value": "query",
        "description": "The language code of the text. Must be in the form of a language code like en-US or fr.",
        "schema": {
            "type": "string"
        },
        "required": true
    }
}
PARAMETER VALUES:
{
    "parameters": {
        "language::query": {
            "value1": "en-US",
            "value2": "fr",
            "value3": "de",
            "value4": "ca",
            "value5": "es",
            "value6": "en-GB",
            "value7": "zh-CN",
            "value8": "ar-DZ",
            "value9": "zu",
            "value10": "th"
        }
    }
}
"""

VALUE_AGENT_BODY_FEWSHOT_PROMPT = """
EXAMPLE 1
SPECIFICATION:
{
    "properties": {
        "personId": {
            "type": "string",
            "description": "The ID of the person. Must be in the form of their name and number. For example John123.",
            "required": true,
            "min_length": 6
        }
    }
}
REQUEST_BODY VALUES:
{
    "request_body": {
        "request_body1": {
            "personId": "John123"
        },
        "request_body2": {
            "personId": "Jane456"
        },
        "request_body3": {
            "personId": "Jack789"
        },
        "request_body4": {
            "personId": "Bob101"
        },
        "request_body5": {
            "personId": "Alice202"
        },
        "request_body6": {
            "personId": "Sam777"
        },
        "request_body7": {
            "personId": "Lily001"
        },
        "request_body8": {
            "personId": "Tom999"
        },
        "request_body9": {
            "personId": "Sara111"
        },
        "request_body10": {
            "personId": "Mike123"
        }
    }
}

EXAMPLE 2
SPECIFICATION:
{
    "properties": {
        "language": {
            "type": "string",
            "description": "The language code of the text. Must be in the form of a language code like en-US or fr.",
            "required": true
        }
    }
}
REQUEST_BODY VALUES:
{
    "request_body": {
        "request_body1": {
            "language": "en-US"
        },
        "request_body2": {
            "language": "fr"
        },
        "request_body3": {
            "language": "de"
        },
        "request_body4": {
            "language": "ca"
        },
        "request_body5": {
            "language": "es"
        },
        "request_body6": {
            "language": "en-GB"
        },
        "request_body7": {
            "language": "zh-CN"
        },
        "request_body8": {
            "language": "ar-DZ"
        },
        "request_body9": {
            "language": "zu"
        },
        "request_body10": {
            "language": "th"
        }
    }
}
"""



ENUM_EXAMPLE_CONSTRAINT_PROMPT = """
Review the provided original SPECIFICATION to generate values. Use the server responses to guide your decision-making process.
You MUST generate values for all parameters and object properties. 
You MUST include all examples or enum values when provided with them in a field or description of each parameter or request body property in your generated values.
"""

#FEWSHOT_REQUEST_BODY_GEN_PROMPT = """
#SUMMARY:
#SCHEMA: {'properties': {'text': {'type': 'string', 'description': "The text to be checked. This or 'data' is required."}, 'data': {'type': 'string', 'description': 'The text to be checked, given as a JSON document that specifies what\'s text and what\'s markup. This or \'text\' is required. Markup will be ignored when looking for errors. Example text: <pre>A &lt;b>test&lt;/b></pre>JSON for the example text: <pre>{"annotation":[\n {"text": "A "},\n {"markup": "&lt;b>"},\n {"text": "test"},\n {"markup": "&lt;/b>"}\n]}</pre> <p>If you have markup that should be interpreted as whitespace, like <tt>&lt;p&gt;</tt> in HTML, you can have it interpreted like this: <pre>{"markup": "&lt;p&gt;", "interpretAs": "\\n\\n"}</pre><p>The \'data\' feature is not limited to HTML or XML, it can be used for any kind of markup.'}, 'language': {'type': 'string', 'description': 'A language code like `en-US`, `de-DE`, `fr`, or `auto` to guess the language automatically (see `preferredVariants` below). For languages with variants (English, German, Portuguese) spell checking will only be activated when you specify the variant, e.g. `en-GB` instead of just `en`.'}, 'altLanguages': {'type': 'string', 'description': 'EXPERIMENTAL: Comma-separated list of language codes to check if a word is not similar to one of the main language (parameter `language`). Unknown words that are similar to a word from the main language will still be considered errors but with type `Hint`. For languages with variants (English, German, Portuguese) you need to specify the variant, e.g. `en-GB` instead of just `en`.'}, 'motherTongue': {'type': 'string', 'description': "A language code of the user's native language, enabling false friends checks for some language pairs."}, 'preferredVariants': {'type': 'string', 'description': 'Comma-separated list of preferred language variants. The language detector used with `language=auto` can detect e.g. English, but it cannot decide whether British English or American English is used. Thus this parameter can be used to specify the preferred variants like `en-GB` and `de-AT`. Only available with `language=auto`.'}, 'enabledRules': {'type': 'string', 'description': 'IDs of rules to be enabled, comma-separated'}, 'disabledRules': {'type': 'string', 'description': 'IDs of rules to be disabled, comma-separated'}, 'enabledCategories': {'type': 'string', 'description': 'IDs of categories to be enabled, comma-separated'}, 'disabledCategories': {'type': 'string', 'description': 'IDs of categories to be disabled, comma-separated'}, 'enabledOnly': {'type': 'boolean', 'description': 'If true, only the rules and categories whose IDs are specified with `enabledRules` or `enabledCategories` are enabled.', 'default': False}}, 'required': ['language']}
#REQUEST_BODY:
#"""

#PARAMETER_REQUIREMENTS_PROMPT = """
#Read each 'description' field for each parameter in the schema to determine restrictions and if the parameter is mutually exclusive with another parameter. Look for key words like 'or' in the description to determine if a parameter is mutually exclusive with another parameter.
#You must generate values for the following parameters unless the description states that the parameter is mutually exclusive with another parameter:
#"""

PARAMETER_REQUIREMENTS_PROMPT = """
Attempt to generate values for the following parameters (attempt the most possible). It is very important to ensure the values are compatible with eachother:
"""

PARAMETER_NECESSITY_PROMPT = """
You MUST generate values for the following parameters using their respective specifications. Ensure that all parameters receive a value without exception:
"""

RETRY_PARAMETER_REQUIREMENTS_PROMPT = """
Attempt to generate values for the following parameters, unless otherwise specified in the failed response. It is very imoprtant to ensure that the values are compatible with eachother.
Remove any parameters that may have caused the failure due to incompatability or convolution:
"""

FAILED_PARAMETER_MATCHINGS_PROMPT = """
You generated the following values for the parameters, but the request was not successful. Here are the values you generated for the parameters:
"""

FAILED_PARAMETER_RESPONSE_PROMPT = """
Here is the response indicating the reason for the operation failure. Attempt to generate new values for the parameters based on the response. You can exclude or alter certain parameters:
"""

FEWSHOT_REQUEST_BODY_GEN_PROMPT = """"""

FEWSHOT_PARAMETER_GEN_PROMPT = """"""

#IDENTIFY_AUTHENTICATION_GEN_PROMPT = """
#Given a summary of an operation and its full specification from the OpenAPI Specification, determine if it consists of any authentication information sent as parameters in either the query or the request body.
#When referring to request body parameters, identify object properties that might be used for authentication.
#Authentication information consists of usernames, passwords, tokens, or any other information that can be made into Bearer tokens, Basic tokens, or API keys.
#If the operation does contain authentication parameters in either the query or request body, indicate the parameters in the following format:
#{
#    "authentication_parameters": {
#        "query_parameters": [list of query parameters],
#        "request_body_parameters": [list of request body parameters]
#    }
#}
#Indicate None for the values of the query_parameters or request_body_parameters if the operation does not contain any authentication parameters.
#"""

IDENTIFY_AUTHENTICATION_GEN_PROMPT = """
Given a summary of an operation and its full specification from the OpenAPI Specification, determine if it consists of any authentication information sent as parameters in either the query or the request body and return your answer in the specified JSON format. 
When referring to request body parameters, identify object properties that might be used for authentication.
Authentication information consists specifically of items related to names or passwords which can be used to create Basic tokens.
If the operation does contain authentication parameters in either the query or request body, indicate the parameters in the following format:
{
    "authentication_parameters": {
        "query_parameters": {
            "username": [parameter name],
            "password": [parameter name]
        },
        "body_parameters": {
            "username": [parameter name],
            "password": [parameter name]
        }
    }
}
It is essential that you label exactly 'None' for the [parameter name] if there are no authentication parameters in the query or request body, respectively, and not any substitute.
Ignore responses. Provide parameters or object properties in the case of request bodies that most closely resemble "account name"/"name"/"username" for "username", and "password" for "password".
If there are multiple parameters that relate to usernames, pick the closest.
Ensure you return a valid JSON object.
"""

FIX_JSON_OBJ = """
There is an issue with the JSON object you generated. When using json.loads() on the JSON string you returned, there is an error.
Here is the incorrect JSON string you returned. Fix the JSON object and return the valid string:
"""

def template_gen_prompt(summary: str, schema: Dict) -> str:
    try:
        schema_str = json.dumps(schema, indent=2)
    except (TypeError, ValueError):
        schema_str = str(schema)
    return f"SUMMARY: {summary}\nSPECIFICATION:\n {schema_str}\n"

def get_value_agent_body_prompt(num: int) -> str:
    return VALUE_AGENT_BODY_PROMPT.replace("[insert number]", str(num))

def get_value_agent_params_prompt(num: int) -> str:
    return VALUE_AGENT_PARAMETERS_PROMPT.replace("[insert number]", str(num))

def get_informed_agent_body_prompt() -> str:
    return INFORMED_VALUE_AGENT_PROMPT.replace("[replace_type]", "request bodies")

def get_informed_agent_params_prompt() -> str:
    return INFORMED_VALUE_AGENT_PROMPT.replace("[replace_type]", "parameters")
