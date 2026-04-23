"""E2E test for json-schema-validation inferlet."""
import json

from conftest import run_inferlet, run_tests


async def test_json_schema_validation(client, args):
    output = await run_inferlet(
        client, "json-schema-validation",
        {"max_tokens": 512},
        timeout=args.timeout,
    )
    assert "Attempt 1/" in output, "Missing attempt header"
    assert "--- Result ---" in output, "Missing result section"

    # On success the inferlet returns the validated JSON as its return value.
    # The return value is the last chunk appended by run_inferlet, so try to
    # parse the tail as JSON.  Even if the LLM fails validation, the output
    # should still contain "Attempt" and "Result" markers.
    if "Schema validation passed!" in output:
        # Extract the JSON returned by the inferlet (last line of output)
        result_section = output.split("--- Result ---")[-1]
        # Find the JSON object in the result section
        for line in result_section.splitlines():
            line = line.strip()
            if line.startswith("{"):
                parsed = json.loads(line)
                assert "name" in parsed, "Missing 'name' field in output"
                break


if __name__ == "__main__":
    run_tests([test_json_schema_validation])
