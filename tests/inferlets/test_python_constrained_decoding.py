"""E2E tests for python-constrained-decoding inferlet.

Verifies that the Python SDK drives a stateful grammar matcher per generated
token (regression test for the constrained-decoding port). Without per-step
matcher driving, output is unconstrained after the first token and would
fail JSON parsing or schema conformance.

Two scenarios:
- ``mode=dict``: ``generate_json(schema=...)`` → parsed dict, returned directly.
- ``mode=pydantic``: ``generate_pydantic(Person)`` → typed instance, returned directly.

Both rely on the bakery wrapper's structured-return support (dict /
pydantic.BaseModel auto-stringified).
"""
import json

from conftest import run_inferlet, run_tests


def _extract_trailing_json(output: str) -> dict | None:
    """Walk back through `{` candidates; return the parsed object that
    consumes the rest of `output`, or None."""
    decoder = json.JSONDecoder()
    start = len(output)
    while True:
        start = output.rfind("{", 0, start)
        if start < 0:
            return None
        try:
            obj, end = decoder.raw_decode(output[start:])
            if end == len(output) - start:
                return obj
        except json.JSONDecodeError:
            continue


def _assert_person_shape(parsed: dict) -> None:
    """Schema conformance — enforced by the grammar; failures here indicate
    the matcher isn't being driven correctly."""
    for field in ("name", "age", "email", "skills"):
        assert field in parsed, f"Missing '{field}' in output: {parsed}"
    assert isinstance(parsed["name"], str) and parsed["name"], "name must be non-empty string"
    assert isinstance(parsed["age"], int) and 0 <= parsed["age"] <= 150, \
        f"age must be int in [0,150], got {parsed['age']!r}"
    assert isinstance(parsed["email"], str), "email must be string"
    assert isinstance(parsed["skills"], list) and len(parsed["skills"]) >= 1, \
        "skills must be non-empty list"
    assert all(isinstance(s, str) for s in parsed["skills"]), \
        "every skill must be a string"


async def test_python_constrained_decoding_dict(client, args):
    """Default 'dict' mode: generate_json(schema=...) returns a parsed dict."""
    output = await run_inferlet(
        client, "python-constrained-decoding",
        {"max_tokens": 512, "mode": "dict"},
        timeout=max(args.timeout, 300),
    )
    assert "Hello " in output, "Missing greeting (typed-field access)"
    assert "Skills:" in output, "Missing skills line"
    assert "[done]" in output, "Missing [done] marker"
    parsed = _extract_trailing_json(output)
    assert parsed is not None, "No trailing JSON object in output"
    _assert_person_shape(parsed)


async def test_python_constrained_decoding_pydantic(client, args):
    """'pydantic' mode: generate_pydantic(Person) returns a typed instance."""
    output = await run_inferlet(
        client, "python-constrained-decoding",
        {"max_tokens": 512, "mode": "pydantic"},
        timeout=max(args.timeout, 300),
    )
    assert "Hello " in output, "Missing greeting (typed-field access)"
    assert "Skills:" in output, "Missing skills line"
    assert "[done]" in output, "Missing [done] marker"
    parsed = _extract_trailing_json(output)
    assert parsed is not None, "No trailing JSON object in output"
    _assert_person_shape(parsed)


if __name__ == "__main__":
    run_tests([test_python_constrained_decoding_dict, test_python_constrained_decoding_pydantic])
