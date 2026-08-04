#!/usr/bin/env python3
"""
Validate that user Python code doesn't import forbidden modules.

This script checks for imports that won't work in the WASM environment:
- Network libraries (requests, urllib3, httpx, aiohttp, etc.)
- Web frameworks (flask, django, fastapi, etc.)
- System libraries (os.*, subprocess, multiprocessing, etc.)
- Database drivers (sqlite3, psycopg2, pymysql, etc.)

Usage:
    python scripts/validate_imports.py <path_to_python_file>
"""

import ast
import sys
from pathlib import Path


# Modules that are forbidden in WASM environment
FORBIDDEN_MODULES = {
    # Network libraries
    "requests",
    "urllib3",
    "httpx",
    "aiohttp",
    "websockets",
    "socket",
    "ssl",
    # Web frameworks
    "flask",
    "django",
    "fastapi",
    "starlette",
    "tornado",
    "bottle",
    # System libraries
    "subprocess",
    "multiprocessing",
    "threading",  # Limited support
    "ctypes",
    "cffi",
    # File system (limited)
    "pathlib",  # Some operations may work
    "shutil",
    "tempfile",
    # Database drivers
    "sqlite3",
    "psycopg2",
    "pymysql",
    "pymongo",
    "redis",
    # Other
    "tkinter",
    "pygame",
    "numpy",  # Native extension
    "pandas",  # Native extension
    "scipy",  # Native extension
}

# Submodules of os that are forbidden
FORBIDDEN_OS_SUBMODULES = {
    "os.system",
    "os.popen",
    "os.spawn",
    "os.fork",
    "os.exec",
}


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to collect imports."""

    def __init__(self) -> None:
        self.imports: list[tuple[str, int]] = []  # (module, line_number)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append((node.module, node.lineno))
            # Also check for specific imports like "from os import system"
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports.append((full_name, node.lineno))
        self.generic_visit(node)


def validate_imports(source_path: Path) -> list[str]:
    """
    Validate imports in a Python source file.

    Returns a list of warning messages for forbidden imports.
    """
    warnings: list[str] = []

    try:
        source = source_path.read_text()
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as e:
        return [f"Syntax error in {source_path}: {e}"]

    visitor = ImportVisitor()
    visitor.visit(tree)

    for module, lineno in visitor.imports:
        # Check exact match
        base_module = module.split(".")[0]
        if base_module in FORBIDDEN_MODULES:
            warnings.append(
                f"{source_path}:{lineno}: Forbidden import '{module}' - "
                f"This module won't work in WASM."
            )
        # Check os submodules
        elif module in FORBIDDEN_OS_SUBMODULES:
            warnings.append(
                f"{source_path}:{lineno}: Forbidden import '{module}' - "
                f"This function won't work in WASM."
            )

    return warnings


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python validate_imports.py <path>", file=sys.stderr)
        print("       path can be a Python file or directory", file=sys.stderr)
        return 1

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 1

    # Collect Python files
    if target.is_file():
        if not target.suffix == ".py":
            print(f"Error: {target} is not a Python file", file=sys.stderr)
            return 1
        files = [target]
    else:
        files = list(target.rglob("*.py"))

    if not files:
        print(f"No Python files found in {target}", file=sys.stderr)
        return 1

    # Validate all files
    all_warnings: list[str] = []
    for f in files:
        # Skip test files and __pycache__
        if "__pycache__" in str(f) or "test" in f.stem.lower():
            continue
        warnings = validate_imports(f)
        all_warnings.extend(warnings)

    if all_warnings:
        print(f"Found {len(all_warnings)} potential issues:\n")
        for warning in all_warnings:
            print(f"  - {warning}")
        print(
            "\nNote: Some of these may work with polyfills or alternatives. "
            "Review each import carefully."
        )
        return 1
    else:
        print(f"Validated {len(files)} file(s) - no forbidden imports found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
