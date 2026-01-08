"""
Runtime wrapper functions for inferlet-core-runtime bindings.
Provides Pythonic API over WIT bindings.
"""

from typing import Any

from wit_world.imports import runtime as _runtime
from .async_utils import await_future

# Track whether set_return has been called
_return_was_set = False


def get_version() -> str:
    """Returns the runtime version string."""
    return _runtime.get_version()


def get_instance_id() -> str:
    """Returns a unique identifier for the running instance."""
    return _runtime.get_instance_id()


def _parse_args(args: list[str]) -> dict[str, str | bool]:
    """
    Parse POSIX-style CLI arguments into a dict.

    Rules:
    1. Flags starting with '-' or '--' become keys.
    2. If a flag is followed by a value (not starting with '-'), it's a key-value pair.
    3. If a flag is followed by another flag or nothing, it's treated as boolean True.
    """
    parsed: dict[str, str | bool] = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith("-"):
            # Remove leading dashes
            key = arg.lstrip("-")

            # Peek at next argument
            next_arg = args[i + 1] if i + 1 < len(args) else None

            # If next arg exists and is NOT a flag, it's the value
            if next_arg and not next_arg.startswith("-"):
                parsed[key] = next_arg
                i += 2
            else:
                # Boolean flag
                parsed[key] = True
                i += 1
        else:
            i += 1

    return parsed


def get_arguments() -> dict[str, Any]:
    """
    Retrieves CLI arguments passed to the inferlet, parsed into a dict.

    Example:
        '--prompt hello --verbose' -> {'prompt': 'hello', 'verbose': True}
    """
    raw_args = _runtime.get_arguments()
    return _parse_args(raw_args)


def set_return(value: str) -> None:
    """Sets the return value for the inferlet."""
    global _return_was_set
    _return_was_set = True
    _runtime.set_return(value)


def was_return_set() -> bool:
    """Check if set_return was called."""
    return _return_was_set


def debug_query(query: str) -> str:
    """
    Executes a debug command and returns the result as a string.

    This is a module-level function that provides debug query capabilities
    without requiring a Queue instance.

    Args:
        query: The debug query string to execute.

    Returns:
        The result of the debug query as a string.
    """
    result = _runtime.debug_query(query)
    return await_future(result, "debug_query result was None")
