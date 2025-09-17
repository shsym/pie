"""
C/C++ Plugin Bindings using ctypes.

Provides interface for loading and interfacing with C/C++ shared libraries
as debug framework plugins.
"""

import ctypes
import ctypes.util
import os
import platform
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from ..models.plugin_definition import PluginDefinition


class CTypesBinding:
    """
    C/C++ plugin binding using ctypes for cross-platform compatibility.
    """

    def __init__(self):
        self.loaded_libraries: Dict[str, ctypes.CDLL] = {}
        self.function_signatures: Dict[str, Dict[str, Any]] = {}

    def load_plugin(self, plugin_def: PluginDefinition) -> bool:
        """
        Load C/C++ shared library plugin.

        Args:
            plugin_def: Plugin definition with library path and signatures

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            lib_path = plugin_def.binary_path
            if not lib_path or not os.path.exists(lib_path):
                return False

            # Load the shared library
            if platform.system() == "Windows":
                lib = ctypes.CDLL(lib_path, winmode=0)
            else:
                lib = ctypes.CDLL(lib_path)

            plugin_id = plugin_def.name if hasattr(plugin_def, 'name') else str(plugin_def.id)
            self.loaded_libraries[plugin_id] = lib

            # Handle function signatures from interface_config
            function_signatures = {}
            if hasattr(plugin_def, 'interface_config') and plugin_def.interface_config:
                function_signatures = plugin_def.interface_config.get('function_signatures', {})

            self.function_signatures[plugin_id] = function_signatures

            # Configure function signatures
            self._configure_function_signatures(plugin_id, lib)

            return True

        except Exception as e:
            print(f"Failed to load C plugin {plugin_id}: {e}")
            return False

    def _configure_function_signatures(self, plugin_id: str, lib: ctypes.CDLL) -> None:
        """Configure ctypes function signatures for type safety."""
        signatures = self.function_signatures.get(plugin_id, {})

        for func_name, sig_info in signatures.items():
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)

                # Set argument types
                if 'argtypes' in sig_info:
                    func.argtypes = self._convert_to_ctypes(sig_info['argtypes'])

                # Set return type
                if 'restype' in sig_info:
                    func.restype = self._convert_to_ctype_single(sig_info['restype'])

    def _convert_to_ctypes(self, type_list: List[str]) -> List[type]:
        """Convert string type names to ctypes types."""
        type_mapping = {
            'int': ctypes.c_int,
            'float': ctypes.c_float,
            'double': ctypes.c_double,
            'char*': ctypes.c_char_p,
            'void*': ctypes.c_void_p,
            'size_t': ctypes.c_size_t,
            'uint32_t': ctypes.c_uint32,
            'int32_t': ctypes.c_int32,
            'uint64_t': ctypes.c_uint64,
            'int64_t': ctypes.c_int64,
        }

        return [type_mapping.get(t, ctypes.c_void_p) for t in type_list]

    def _convert_to_ctype_single(self, type_name: str) -> type:
        """Convert single type name to ctypes type."""
        type_mapping = {
            'int': ctypes.c_int,
            'float': ctypes.c_float,
            'double': ctypes.c_double,
            'char*': ctypes.c_char_p,
            'void*': ctypes.c_void_p,
            'void': None,
            'size_t': ctypes.c_size_t,
        }

        return type_mapping.get(type_name, ctypes.c_void_p)

    def call_function(self, plugin_id: str, function_name: str, *args) -> Any:
        """
        Call function from loaded C/C++ plugin.

        Args:
            plugin_id: ID of the loaded plugin
            function_name: Name of the function to call
            *args: Arguments to pass to the function

        Returns:
            Function return value

        Raises:
            ValueError: If plugin or function not found
        """
        if plugin_id not in self.loaded_libraries:
            raise ValueError(f"Plugin {plugin_id} not loaded")

        lib = self.loaded_libraries[plugin_id]

        if not hasattr(lib, function_name):
            raise ValueError(f"Function {function_name} not found in plugin {plugin_id}")

        func = getattr(lib, function_name)
        return func(*args)

    def get_function_pointer(self, plugin_id: str, function_name: str) -> Optional[Callable]:
        """Get function pointer for callback usage."""
        if plugin_id not in self.loaded_libraries:
            return None

        lib = self.loaded_libraries[plugin_id]
        if not hasattr(lib, function_name):
            return None

        return getattr(lib, function_name)

    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload plugin library.

        Args:
            plugin_id: ID of the plugin to unload

        Returns:
            True if successfully unloaded
        """
        if plugin_id in self.loaded_libraries:
            # Note: ctypes doesn't provide explicit unloading
            # Library will be unloaded when Python process exits
            del self.loaded_libraries[plugin_id]
            if plugin_id in self.function_signatures:
                del self.function_signatures[plugin_id]
            return True
        return False

    def list_loaded_plugins(self) -> List[str]:
        """Get list of currently loaded plugin IDs."""
        return list(self.loaded_libraries.keys())

    def get_plugin_functions(self, plugin_id: str) -> List[str]:
        """Get list of available functions in a plugin."""
        if plugin_id not in self.function_signatures:
            return []
        return list(self.function_signatures[plugin_id].keys())


def detect_c_library(path: str) -> bool:
    """
    Detect if file is a C/C++ shared library.

    Args:
        path: Path to the file

    Returns:
        True if it's a recognized C/C++ shared library
    """
    if not os.path.exists(path):
        return False

    # Check file extension
    extensions = {'.so', '.dll', '.dylib', '.a'}
    path_obj = Path(path)

    return path_obj.suffix.lower() in extensions


def create_c_plugin_definition(
    plugin_name: str,
    library_path: str,
    functions: Dict[str, Dict[str, Any]]
) -> PluginDefinition:
    """
    Create plugin definition for C/C++ library.

    Args:
        plugin_name: Name for the plugin
        library_path: Path to the shared library
        functions: Function signatures dictionary

    Returns:
        PluginDefinition configured for C/C++ plugin
    """
    # Create temporary source file for PluginDefinition requirement
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False)
    temp_file.write("// Temporary C source for ctypes binding")
    temp_file.close()

    return PluginDefinition(
        name=plugin_name,
        version="1.0.0",
        target_platform="cpp",
        source_files=[temp_file.name],
        compile_config={"target_platform": "cpp"},
        interface_config={
            'function_signatures': functions,
            'language': 'C/C++',
            'binding_type': 'ctypes',
            'platform': platform.system(),
        },
        binary_path=library_path,
        is_compiled=True
    )