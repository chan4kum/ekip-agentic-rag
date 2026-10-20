import pytest
from ekip_tools.schema import SchemaError, check_schema, validate_arguments

SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 2, "maxLength": 10, "pattern": "[a-z]+"},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "n": {"type": "integer", "minimum": 1, "maximum": 5},
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "flag": {"type": "boolean"},
    },
}


def test_valid_arguments_pass() -> None:
    assert (
        validate_arguments(SCHEMA, {"name": "abc", "kind": "a", "n": 3, "x": 0.5, "flag": True})
        == []
    )


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "name: required"),
        ({"name": "abc", "extra": 1}, "extra: unknown field"),
        ({"name": "a"}, "too short"),
        ({"name": "a" * 11}, "too long"),
        ({"name": "ABC"}, "wrong format"),
        ({"name": 5}, "must be a string"),
        ({"name": "abc", "kind": "z"}, "not an allowed value"),
        ({"name": "abc", "n": 6}, "out of range"),
        ({"name": "abc", "n": True}, "must be a number"),
        ({"name": "abc", "n": 2.5}, "must be an integer"),
        ({"name": "abc", "flag": "yes"}, "must be a boolean"),
    ],
)
def test_invalid_arguments_are_reported(arguments: dict[str, object], fragment: str) -> None:
    assert any(fragment in p for p in validate_arguments(SCHEMA, arguments))


def test_errors_never_echo_values() -> None:
    payload = "sk-SECRET-VALUE"
    problems = validate_arguments(SCHEMA, {"name": payload, "evil\nfield": payload})
    assert problems
    assert all(payload not in p for p in problems)
    assert all("\n" not in p for p in problems)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array"},
        {"type": "object", "properties": {"a": {"type": "string"}}},  # unbounded string
        {"type": "object", "properties": {"a": {"type": "integer"}}},  # unbounded number
        {"type": "object", "properties": {"a": {"type": "object"}}},
        {"type": "object", "properties": {"a": {"type": "string", "maxLength": 5, "format": "x"}}},
        {"type": "object", "properties": {"a": {"type": "string", "maxLength": 5, "pattern": "("}}},
        {"type": "object", "required": ["missing"], "properties": {}},
        {"type": "object", "$ref": "#/x"},
    ],
)
def test_unsupported_schemas_are_rejected_at_load(schema: dict[str, object]) -> None:
    with pytest.raises(SchemaError):
        check_schema(schema)
