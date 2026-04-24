"""E2E test for json-schema-validation inferlet."""
import json

from conftest import run_inferlet, run_tests


async def test_json_schema_validation(client, args):
    output = await run_inferlet(
        client, "json-schema-validation",
        ["--max-tokens", "512"],
        timeout=max(args.timeout, 300),
    )
    assert "Attempt 1/" in output, "Missing attempt header"
    assert "--- Result ---" in output, "Missing result section"

    # On success the inferlet returns the validated JSON as its return value.
    if "Schema validation passed!" in output:
        result_section = output.split("--- Result ---")[-1]
        for line in result_section.splitlines():
            line = line.strip()
            if line.startswith("{"):
                parsed = json.loads(line)
                assert "name" in parsed, "Missing 'name' field in output"
                break


if __name__ == "__main__":
    run_tests([test_json_schema_validation])
