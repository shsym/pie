"""E2E test for json-schema-validation inferlet."""
import json

from conftest import run_inferlet, run_tests


async def test_json_schema_validation(client, args):
    output = await run_inferlet(
        client, "json-schema-validation",
        {"max_tokens": 1024, "max_retries": 1},
        timeout=max(args.timeout, 300),
    )
    assert "Attempt 1/" in output, "Missing attempt header"
    assert "--- Result ---" in output, "Missing result section"
    assert "Schema validation passed!" in output, "Grammar-constrained JSON failed to validate"

    # The inferlet returns the validated JSON as its return value, which
    # `run_inferlet` appends at the very end of `output`. Walk backwards
    # through "{" candidates and pick the one that parses to a JSON object
    # consuming the rest of the output (= the trailing return-value JSON).
    if True:
        decoder = json.JSONDecoder()
        parsed = None
        start = len(output)
        while True:
            start = output.rfind("{", 0, start)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(output[start:])
                if end == len(output) - start:
                    parsed = obj
                    break
            except json.JSONDecodeError:
                continue
        assert parsed is not None, "No trailing JSON object in output"
        for field in ("name", "age", "email", "skills", "address"):
            assert field in parsed, f"Missing '{field}' field in output"


if __name__ == "__main__":
    run_tests([test_json_schema_validation])
