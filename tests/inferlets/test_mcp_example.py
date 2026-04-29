"""E2E test for the mcp-example inferlet.

Spawns a self-contained stdio MCP server (`_mcp_demo_server.py`), registers
it with the running Pie engine through the Python client's bridge, runs the
inferlet, and asserts the output round-trips through every layer.
"""
import sys
from pathlib import Path

from conftest import run_inferlet, run_tests

DEMO_SERVER = Path(__file__).parent / "_mcp_demo_server.py"


async def _register_demo(client) -> None:
    """Register the demo MCP server under the name 'demo' if not already."""
    # Tests are independent — each registers fresh. If a prior test in the
    # same client already registered, register_mcp_server raises; ignore.
    try:
        await client.register_mcp_server(
            name="demo",
            transport="stdio",
            command=sys.executable,
            args=[str(DEMO_SERVER)],
        )
    except Exception as e:
        if "already registered" not in str(e):
            raise


async def test_mcp_echo(client, args):
    await _register_demo(client)
    out = await run_inferlet(
        client, "mcp-example",
        {"server": "demo", "tool": "echo", "text": "ping"},
        timeout=args.timeout,
    )
    assert "available servers: " in out, f"Missing servers list in:\n{out}"
    assert "\"demo\"" in out, f"'demo' not listed in:\n{out}"
    assert "result: ping" in out, f"Echo result not found in:\n{out}"


async def test_mcp_list_tools(client, args):
    await _register_demo(client)
    out = await run_inferlet(
        client, "mcp-example",
        {"server": "demo", "tool": "echo", "text": "x"},
        timeout=args.timeout,
    )
    # tools_json is printed on a `tools: ...` line.
    assert "\"echo\"" in out and "\"add\"" in out and "\"fail\"" in out, \
        f"Expected all three tools to appear in tools_json:\n{out}"


async def test_mcp_unknown_server(client, args):
    # Don't register — expect a clean error from the inferlet's available_servers
    # check (so we exercise the not-registered path without spawning the bridge).
    try:
        await run_inferlet(
            client, "mcp-example",
            {"server": "nope-not-a-server"},
            timeout=args.timeout,
        )
    except RuntimeError as e:
        assert "not registered" in str(e), f"Unexpected error message: {e}"
        return
    raise AssertionError("expected error for unregistered MCP server")


async def test_mcp_tool_error(client, args):
    """isError=true at the MCP tool layer must surface as an inferlet error.

    Regression guard: an earlier version of the runtime ignored isError and
    returned the content array as a normal success.
    """
    await _register_demo(client)
    try:
        await run_inferlet(
            client, "mcp-example",
            {"server": "demo", "tool": "tool_error", "text": "n/a"},
            timeout=args.timeout,
        )
    except RuntimeError as e:
        assert "tool reported failure" in str(e), f"Unexpected error message: {e}"
        return
    raise AssertionError("expected isError=true to surface as an error")


if __name__ == "__main__":
    run_tests([test_mcp_echo, test_mcp_list_tools, test_mcp_unknown_server, test_mcp_tool_error])
