"""E2E test for template-generation inferlet."""
from conftest import run_inferlet, run_tests


async def test_template_generation(client, args):
    output = await run_inferlet(
        client, "template-generation",
        {"max_tokens": 2048, "max_retries": 1},
        timeout=max(args.timeout, 300),
    )
    assert "Attempt 1/" in output, "Missing attempt header"
    assert "--- Result ---" in output, "Missing result section"
    assert "Rendered successfully." in output, "Template generation did not succeed"
    assert "PRODUCT ANNOUNCEMENT" in output, "Missing rendered template header"


if __name__ == "__main__":
    run_tests([test_template_generation])
