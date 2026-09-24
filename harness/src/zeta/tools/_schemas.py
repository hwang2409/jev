"""Normalize tool schemas and validate tool arguments."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any


def normalize_schema(
    schema: Mapping[str, Any] | None,
    *,
    validate_definition: bool = True,
) -> dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping):
        raise TypeError("tool parameter schema must be an object")
    try:
        normalized = copy.deepcopy(dict(schema))
    except Exception as exc:
        raise ValueError("schema must contain JSON data") from exc
    if validate_definition:
        validate_schema_definition(normalized, "schema")
    validate_json_data(normalized, "schema")
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("schema must contain JSON data") from exc
    return normalized


def validate_json_data(value: Any, path: str) -> None:
    if value is None or type(value) in {bool, float, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            validate_json_data(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"schema must contain JSON data at {path}")
            validate_json_data(item, f"{path}.{key}")
        return
    raise ValueError(f"schema must contain JSON data at {path}")


def validate_arguments(arguments: object, schema: Mapping[str, Any]) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    validate_finite_numbers(arguments, "arguments")
    validate_schema(arguments, schema, "arguments")
    return dict(arguments)


def coerce_arguments(arguments: object) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    validate_finite_numbers(arguments, "arguments")
    return dict(arguments)


def validate_finite_numbers(value: Any, path: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite_numbers(child, f"{path}[{index}]")


def validate_schema(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not matches_type(value, expected_type):
        raise ValueError(f"{path} must be {expected_type}")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if "const" in schema and not schema_equal(value, schema["const"]):
        raise ValueError(f"{path} must equal the declared constant")
    if "enum" in schema and not any(
        schema_equal(value, option) for option in schema["enum"]
    ):
        raise ValueError(f"{path} is not an allowed value")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is too long")
    if type(value) in {int, float} and type(value) is not bool:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path} is not above the exclusive minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above the maximum")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} has unexpected properties: {', '.join(extra)}")
        for key, child_schema in properties.items():
            if key in value:
                validate_schema(value[key], child_schema, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                validate_schema(item, item_schema, f"{path}[{index}]")


def matches_type(value: Any, expected: object) -> bool:
    if type(expected) is list:
        return any(matches_type(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in {int, float}
    if expected == "null":
        return value is None
    return False


def valid_schema_types(expected: object) -> bool:
    if type(expected) is str:
        return expected in SCHEMA_TYPES
    return (
        type(expected) is list
        and bool(expected)
        and all(type(item) is str and item in SCHEMA_TYPES for item in expected)
    )


def schema_equal(left: Any, right: Any) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            set(left) == set(right)
            and all(schema_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            schema_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


SCHEMA_KEYS = {
    "description",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minLength",
    "maxLength",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "minItems",
    "maxItems",
    "enum",
    "const",
}
SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def validate_schema_definition(schema: Mapping[str, Any], path: str) -> None:
    unsupported = set(schema) - SCHEMA_KEYS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported schema keywords at {path}: {names}")
    expected_type = schema.get("type")
    if expected_type is not None and not valid_schema_types(expected_type):
        raise ValueError(f"unsupported schema type at {path}")
    description = schema.get("description")
    if description is not None and type(description) is not str:
        raise ValueError(f"schema description must be a string at {path}")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError(f"schema properties must be an object at {path}")
        for name, child in properties.items():
            if type(name) is not str or not isinstance(child, Mapping):
                raise ValueError(f"invalid schema property at {path}")
            validate_schema_definition(child, f"{path}.{name}")
    required = schema.get("required")
    if required is not None and (
        type(required) is not list or any(type(name) is not str for name in required)
    ):
        raise ValueError(f"schema required must be a string array at {path}")
    additional = schema.get("additionalProperties")
    if additional is not None and type(additional) is not bool:
        raise ValueError(f"schema additionalProperties must be boolean at {path}")
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"schema items must be an object at {path}")
        validate_schema_definition(items, f"{path}.items")
    enum = schema.get("enum")
    if enum is not None and type(enum) is not list:
        raise ValueError(f"schema enum must be an array at {path}")
    for key in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    ):
        value = schema.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"schema {key} must be a nonnegative integer at {path}")
    for key in ("minimum", "exclusiveMinimum", "maximum"):
        value = schema.get(key)
        if value is not None and (type(value) not in {int, float} or type(value) is bool):
            raise ValueError(f"schema {key} must be numeric at {path}")
    if expected_type == "object" and schema.get("items") is not None:
        raise ValueError(f"schema items is not valid for an object at {path}")
    if expected_type != "object" and schema.get("properties") is not None:
        raise ValueError(f"schema properties is only valid for an object at {path}")
    if expected_type != "array" and schema.get("items") is not None:
        raise ValueError(f"schema items is only valid for an array at {path}")


_normalize_schema = normalize_schema
_validate_json_data = validate_json_data
_validate_arguments = validate_arguments
_coerce_arguments = coerce_arguments
_validate_finite_numbers = validate_finite_numbers
_validate_schema = validate_schema
_matches_type = matches_type
_valid_schema_types = valid_schema_types
_schema_equal = schema_equal
_SCHEMA_KEYS = SCHEMA_KEYS
_SCHEMA_TYPES = SCHEMA_TYPES
_validate_schema_definition = validate_schema_definition
