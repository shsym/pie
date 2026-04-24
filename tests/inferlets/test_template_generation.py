"""E2E test for template-generation inferlet."""
from conftest import run_inferlet, run_tests


async def test_template_generation(client, args):
    output = await run_inferlet(
        client, "template-generation",
        ["--max-tokens", "1024"],
        timeout=max(args.timeout, 300),
    )
    assert "Attempt 1/" in output, "Missing attempt header"
    assert "--- Result ---" in output, "Missing result section"

    # On success the rendered template includes "PRODUCT ANNOUNCEMENT"
    if "Rendered successfully." in output:
        assert "PRODUCT ANNOUNCEMENT" in output, "Missing rendered template header"


if __name__ == "__main__":
    run_tests([test_template_generation])
