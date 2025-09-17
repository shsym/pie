"""
Plugin registry service for managing debug framework plugins.

This service handles plugin discovery, registration, compilation, and lifecycle
management for different backend validation plugins (Metal, CUDA, C++).
"""

import os
import json
import hashlib
import platform
import subprocess
import shutil
import sys
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import ctypes


class PluginRegistry:
    """
    Central registry for managing debug framework plugins.

    Handles plugin discovery, registration, compilation, lifecycle management,
    and dependency resolution for different backend validation plugins.
    """

    SUPPORTED_BACKENDS = ["metal", "cuda", "cpp", "python"]
    REQUIRED_FIELDS = ["name", "version", "backend_type", "supported_operations", "source_file"]

    def __init__(self, plugin_directory: str = None, cache_directory: str = None):
        """
        Initialize the plugin registry.

        Args:
            plugin_directory: Directory containing plugin source files
            cache_directory: Directory for cached compiled binaries
        """
        self.plugin_directory = plugin_directory or "/tmp/debug_plugins"
        self.cache_directory = cache_directory or "/tmp/plugin_cache"
        self.registered_plugins: Dict[str, Dict[str, Any]] = {}
        self.loaded_plugins: Dict[str, Dict[str, Any]] = {}
        self.compilation_cache: Dict[str, str] = {}

        # Ensure directories exist
        os.makedirs(self.plugin_directory, exist_ok=True)
        os.makedirs(self.cache_directory, exist_ok=True)

    def discover_plugins(self) -> List[Dict[str, Any]]:
        """
        Discover plugin files in the plugin directory.

        Returns:
            List of discovered plugin metadata dictionaries
        """
        discovered_plugins = []

        if not os.path.exists(self.plugin_directory):
            return discovered_plugins

        try:
            for filename in os.listdir(self.plugin_directory):
                if os.path.isfile(os.path.join(self.plugin_directory, filename)):
                    # Skip non-plugin files like README.md
                    if filename.endswith('.md'):
                        continue

                    # Extract plugin name from filename (without extension)
                    plugin_name = os.path.splitext(filename)[0]

                    # Determine type based on extension
                    plugin_type = self._get_plugin_type_from_extension(filename)

                    plugin_metadata = {
                        "name": plugin_name,
                        "source_file": os.path.join(self.plugin_directory, filename),
                        "type": plugin_type
                    }

                    discovered_plugins.append(plugin_metadata)
        except Exception:
            pass

        return discovered_plugins

    def _get_plugin_type_from_extension(self, filename: str) -> str:
        """Determine plugin type from file extension."""
        ext = os.path.splitext(filename)[1].lower()

        if ext in ['.cpp', '.cc', '.cxx']:
            return 'cpp'
        elif ext in ['.cu']:
            return 'cuda'
        elif ext in ['.py']:
            return 'python'
        elif ext in ['.json']:
            return 'config'
        else:
            return 'unknown'

    def register_plugin(self, plugin_metadata: Dict[str, Any]) -> str:
        """
        Register a plugin with metadata validation.

        Args:
            plugin_metadata: Plugin metadata dictionary

        Returns:
            str: Plugin ID

        Raises:
            ValueError: If metadata validation fails
        """
        # Validate required fields
        for field in self.REQUIRED_FIELDS:
            if field not in plugin_metadata:
                raise ValueError(f"Missing required field: {field}")

        # Validate backend type
        backend_type = plugin_metadata.get("backend_type")
        if backend_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend_type: {backend_type}")

        # Validate version format
        version = plugin_metadata.get("version")
        if not self._validate_version_format(version):
            raise ValueError(f"Invalid version format: {version}")

        # Generate unique plugin ID
        plugin_id = str(uuid.uuid4())

        # Store plugin metadata
        self.registered_plugins[plugin_id] = plugin_metadata.copy()

        return plugin_id

    def is_registered(self, plugin_name_or_id: str) -> bool:
        """
        Check if a plugin is registered by name or ID.

        Args:
            plugin_name_or_id: Plugin name or plugin ID

        Returns:
            bool: True if plugin is registered
        """
        # Check if it's a plugin ID (UUID format)
        if plugin_name_or_id in self.registered_plugins:
            return True

        # Check if it's a plugin name
        for plugin_metadata in self.registered_plugins.values():
            if plugin_metadata.get("name") == plugin_name_or_id:
                return True

        return False

    def _validate_version_format(self, version: str) -> bool:
        """Validate version follows semantic versioning format."""
        if not version:
            return False

        # Basic semver pattern: major.minor.patch
        pattern = r'^\d+\.\d+\.\d+$'
        return re.match(pattern, version) is not None

    def compile_plugin(self, plugin_id: str) -> str:
        """
        Compile a plugin and return the compiled binary path.

        Delegates to the build toolchain (Make/CMake) for caching and dependency tracking.

        Args:
            plugin_id: Plugin ID to compile

        Returns:
            str: Path to compiled binary

        Raises:
            RuntimeError: If compilation fails
        """
        if plugin_id not in self.registered_plugins:
            raise ValueError(f"Plugin not found: {plugin_id}")

        plugin_metadata = self.registered_plugins[plugin_id]
        backend_type = plugin_metadata.get("backend_type")

        # Delegate to build toolchain for compilation and caching
        if backend_type == "metal":
            return self._compile_metal_plugin(plugin_id, plugin_metadata)
        elif backend_type == "cuda":
            return self._compile_cuda_plugin(plugin_id, plugin_metadata)
        elif backend_type == "cpp":
            return self._compile_cpp_plugin(plugin_id, plugin_metadata)
        else:
            raise RuntimeError(f"Compilation not supported for backend: {backend_type}")

    def _compile_metal_plugin(self, plugin_id: str, metadata: Dict[str, Any]) -> str:
        """Compile Metal plugin."""
        source_file = metadata.get("source_file")
        compilation_flags = metadata.get("compilation_flags", ["-std=c++17"])

        output_path = os.path.join(self.cache_directory, f"{plugin_id}.dylib")

        compile_cmd = ["xcrun", "clang++"] + compilation_flags + ["-shared", source_file, "-o", output_path]

        # Add Metal framework
        if "-framework" not in compile_cmd:
            compile_cmd.extend(["-framework", "Metal"])

        result = subprocess.run(compile_cmd, capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f"Plugin compilation failed: {result.stderr.decode()}")

        return output_path

    def _compile_cuda_plugin(self, plugin_id: str, metadata: Dict[str, Any]) -> str:
        """Compile CUDA plugin."""
        source_file = metadata.get("source_file")
        compilation_flags = metadata.get("compilation_flags", ["-O3"])
        host_compiler_flags = metadata.get("host_compiler_flags", [])

        output_path = os.path.join(self.cache_directory, f"{plugin_id}.so")

        compile_cmd = ["nvcc"] + compilation_flags

        # Add host compiler flags with -Xcompiler
        for flag in host_compiler_flags:
            compile_cmd.extend(["-Xcompiler", flag])

        compile_cmd.extend(["-shared", source_file, "-o", output_path])

        result = subprocess.run(compile_cmd, capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f"Plugin compilation failed: {result.stderr.decode()}")

        return output_path

    def _compile_cpp_plugin(self, plugin_id: str, metadata: Dict[str, Any]) -> str:
        """Compile C++ plugin."""
        source_file = metadata.get("source_file")
        compilation_flags = metadata.get("compilation_flags", ["-std=c++17", "-O3"])

        output_path = os.path.join(self.cache_directory, f"{plugin_id}.so")

        compile_cmd = ["g++"] + compilation_flags + ["-shared", "-fPIC", source_file, "-o", output_path]

        result = subprocess.run(compile_cmd, capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f"Plugin compilation failed: {result.stderr.decode()}")

        return output_path

    def load_plugin(self, plugin_id: str, binary_path: str) -> Any:
        """
        Load a compiled plugin binary.

        Args:
            plugin_id: Plugin ID
            binary_path: Path to compiled binary

        Returns:
            Loaded plugin library

        Raises:
            ValueError: If interface validation fails
        """
        if plugin_id in self.loaded_plugins:
            return self.loaded_plugins[plugin_id]["library"]

        plugin_metadata = self.registered_plugins.get(plugin_id)
        if not plugin_metadata:
            raise ValueError(f"Plugin not found: {plugin_id}")

        # Load the library
        library = ctypes.CDLL(binary_path)

        # Validate interface version if specified
        interface_version = plugin_metadata.get("interface_version")
        if interface_version:
            try:
                lib_version = library.get_interface_version().decode('utf-8')
                if lib_version != interface_version:
                    raise ValueError(f"Interface version mismatch: expected {interface_version}, got {lib_version}")
            except AttributeError:
                # Library doesn't have interface version function
                pass

        # Validate interface
        if hasattr(library, 'validate_interface'):
            if not library.validate_interface():
                raise ValueError("Interface validation failed")

        # Store loaded plugin
        self.loaded_plugins[plugin_id] = {
            "library": library,
            "metadata": plugin_metadata
        }

        return library

    def execute_operation(self, plugin_id: str, operation: str, params: Dict[str, Any]) -> Any:
        """
        Execute an operation on a loaded plugin.

        Args:
            plugin_id: Plugin ID
            operation: Operation name
            params: Operation parameters

        Returns:
            Operation result
        """
        if plugin_id not in self.loaded_plugins:
            raise ValueError(f"Plugin not loaded: {plugin_id}")

        library = self.loaded_plugins[plugin_id]["library"]

        # Get the operation function
        if hasattr(library, operation):
            operation_func = getattr(library, operation)
            return operation_func(**params)
        else:
            raise ValueError(f"Operation not found: {operation}")

    def resolve_dependencies(self, plugin_ids: List[str]) -> List[str]:
        """
        Resolve plugin dependencies and return load order.

        Args:
            plugin_ids: List of plugin IDs to resolve

        Returns:
            List of plugin IDs in dependency order

        Raises:
            ValueError: If circular dependency detected
        """
        # Build dependency graph
        dependencies = {}
        for plugin_id in plugin_ids:
            plugin_metadata = self.registered_plugins.get(plugin_id)
            if plugin_metadata:
                deps = plugin_metadata.get("dependencies", [])
                dependencies[plugin_id] = deps

        # Topological sort with cycle detection
        visited = set()
        visiting = set()
        result = []

        def visit(plugin_id):
            if plugin_id in visiting:
                # Circular dependency detected
                cycle_path = list(visiting) + [plugin_id]
                cycle_names = []
                for pid in cycle_path:
                    metadata = self.registered_plugins.get(pid, {})
                    cycle_names.append(metadata.get("name", pid))
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle_names)}")

            if plugin_id in visited:
                return

            visiting.add(plugin_id)

            # Visit dependencies first
            plugin_deps = dependencies.get(plugin_id, [])
            for dep_name in plugin_deps:
                # Find plugin ID by name
                dep_id = self._find_plugin_id_by_name(dep_name)
                if dep_id and dep_id in plugin_ids:
                    visit(dep_id)

            visiting.remove(plugin_id)
            visited.add(plugin_id)
            result.append(plugin_id)

        for plugin_id in plugin_ids:
            if plugin_id not in visited:
                visit(plugin_id)

        return result

    def _find_plugin_id_by_name(self, plugin_name: str) -> Optional[str]:
        """Find plugin ID by plugin name."""
        for plugin_id, metadata in self.registered_plugins.items():
            if metadata.get("name") == plugin_name:
                return plugin_id
        return None

    def unload_plugin(self, plugin_id: str):
        """
        Unload a plugin from memory.

        Args:
            plugin_id: Plugin ID to unload
        """
        if plugin_id in self.loaded_plugins:
            loaded_plugin = self.loaded_plugins[plugin_id]
            library = loaded_plugin["library"]

            # Call cleanup if available
            if hasattr(library, 'cleanup'):
                library.cleanup()

            # Remove from loaded plugins
            del self.loaded_plugins[plugin_id]

    def get_platform_compatible_plugins(self, operation: str) -> List[Dict[str, Any]]:
        """
        Get plugins compatible with current platform for given operation.

        Args:
            operation: Operation name

        Returns:
            List of compatible plugin metadata
        """
        compatible = []
        current_system = platform.system()

        for plugin_id, metadata in self.registered_plugins.items():
            # Check operation support
            supported_ops = metadata.get("supported_operations", [])
            if operation not in supported_ops:
                continue

            # Check platform compatibility
            plugin_platform = metadata.get("platform")
            backend_type = metadata.get("backend_type")

            if plugin_platform:
                # Explicit platform specified
                if plugin_platform.lower() != current_system.lower():
                    continue
            elif backend_type == "metal":
                # Metal only works on macOS
                if current_system != "Darwin":
                    continue

            compatible.append(metadata)

        return compatible

    def save_state(self) -> Dict[str, Any]:
        """
        Save current registry state.

        Returns:
            Registry state dictionary
        """
        return {
            "registered_plugins": self.registered_plugins.copy(),
            "compilation_cache": self.compilation_cache.copy()
        }

    def load_state(self, state: Dict[str, Any]):
        """
        Load registry state.

        Args:
            state: Registry state dictionary
        """
        if "registered_plugins" in state:
            self.registered_plugins = state["registered_plugins"].copy()

        if "compilation_cache" in state:
            self.compilation_cache = state["compilation_cache"].copy()

    def get_latest_version(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get latest version of a plugin by name.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin metadata with latest version
        """
        matching_plugins = []

        for plugin_id, metadata in self.registered_plugins.items():
            if metadata.get("name") == plugin_name:
                matching_plugins.append({
                    "id": plugin_id,
                    "version": metadata.get("version"),
                    **metadata
                })

        if not matching_plugins:
            raise ValueError(f"No plugin found with name: {plugin_name}")

        # Sort by version (simple string comparison for semver)
        matching_plugins.sort(key=lambda x: x.get("version", "0.0.0"), reverse=True)

        return matching_plugins[0]

    def get_plugin_by_version(self, plugin_name: str, version: str) -> Dict[str, Any]:
        """
        Get specific version of a plugin by name.

        Args:
            plugin_name: Plugin name
            version: Plugin version

        Returns:
            Plugin metadata for specific version
        """
        for plugin_id, metadata in self.registered_plugins.items():
            if (metadata.get("name") == plugin_name and
                metadata.get("version") == version):
                return {
                    "id": plugin_id,
                    **metadata
                }

        raise ValueError(f"No plugin found with name: {plugin_name}, version: {version}")