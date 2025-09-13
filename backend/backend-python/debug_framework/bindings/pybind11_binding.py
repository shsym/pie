"""
C++ Plugin Bindings using pybind11.

Provides interface for loading and interfacing with C++ libraries
compiled with pybind11 Python bindings.
"""

import importlib
import importlib.util
import os
import sys
from typing import Dict, Any, List, Optional, Callable, Type
from pathlib import Path

from ..models.plugin_definition import PluginDefinition


class Pybind11Binding:
    """
    C++ plugin binding using pybind11 for native Python extension modules.
    """

    def __init__(self):
        self.loaded_modules: Dict[str, Any] = {}
        self.function_signatures: Dict[str, Dict[str, Any]] = {}

    def load_plugin(self, plugin_def: PluginDefinition) -> bool:
        """
        Load pybind11 C++ extension module plugin.

        Args:
            plugin_def: Plugin definition with module path and signatures

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            module_path = plugin_def.binary_path
            if not module_path or not os.path.exists(module_path):
                return False

            # Load the pybind11 extension module
            module = self._load_extension_module(plugin_def.plugin_id, module_path)
            if not module:
                return False

            self.loaded_modules[plugin_def.plugin_id] = module
            self.function_signatures[plugin_def.plugin_id] = plugin_def.function_signatures or {}

            return True

        except Exception as e:
            print(f"Failed to load pybind11 plugin {plugin_def.plugin_id}: {e}")
            return False

    def _load_extension_module(self, plugin_id: str, module_path: str) -> Optional[Any]:
        """Load Python extension module from file path."""
        try:
            # Create module spec from file path
            spec = importlib.util.spec_from_file_location(plugin_id, module_path)
            if not spec or not spec.loader:
                return None

            # Create and load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            print(f"Failed to load extension module {plugin_id}: {e}")
            return None

    def call_function(self, plugin_id: str, function_name: str, *args, **kwargs) -> Any:
        """
        Call function from loaded pybind11 plugin.

        Args:
            plugin_id: ID of the loaded plugin
            function_name: Name of the function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function return value

        Raises:
            ValueError: If plugin or function not found
        """
        if plugin_id not in self.loaded_modules:
            raise ValueError(f"Plugin {plugin_id} not loaded")

        module = self.loaded_modules[plugin_id]

        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in plugin {plugin_id}")

        func = getattr(module, function_name)
        return func(*args, **kwargs)

    def get_class(self, plugin_id: str, class_name: str) -> Optional[Type]:
        """
        Get class from loaded pybind11 plugin.

        Args:
            plugin_id: ID of the loaded plugin
            class_name: Name of the class to get

        Returns:
            Class object or None
        """
        if plugin_id not in self.loaded_modules:
            return None

        module = self.loaded_modules[plugin_id]

        if not hasattr(module, class_name):
            return None

        return getattr(module, class_name)

    def create_instance(self, plugin_id: str, class_name: str, *args, **kwargs) -> Any:
        """
        Create instance of class from loaded pybind11 plugin.

        Args:
            plugin_id: ID of the loaded plugin
            class_name: Name of the class to instantiate
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor

        Returns:
            Class instance

        Raises:
            ValueError: If plugin or class not found
        """
        cls = self.get_class(plugin_id, class_name)
        if not cls:
            raise ValueError(f"Class {class_name} not found in plugin {plugin_id}")

        return cls(*args, **kwargs)

    def call_method(
        self,
        instance: Any,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call method on pybind11 class instance.

        Args:
            instance: Class instance
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Method return value

        Raises:
            ValueError: If method not found
        """
        if not hasattr(instance, method_name):
            raise ValueError(f"Method {method_name} not found on instance")

        method = getattr(instance, method_name)
        return method(*args, **kwargs)

    def get_module_attributes(self, plugin_id: str) -> List[str]:
        """
        Get list of available attributes in loaded module.

        Args:
            plugin_id: ID of the loaded plugin

        Returns:
            List of attribute names
        """
        if plugin_id not in self.loaded_modules:
            return []

        module = self.loaded_modules[plugin_id]
        return [attr for attr in dir(module) if not attr.startswith('_')]

    def get_function_signature(self, plugin_id: str, function_name: str) -> Optional[str]:
        """
        Get function signature documentation.

        Args:
            plugin_id: ID of the loaded plugin
            function_name: Name of the function

        Returns:
            Function signature string or None
        """
        if plugin_id not in self.loaded_modules:
            return None

        module = self.loaded_modules[plugin_id]

        if not hasattr(module, function_name):
            return None

        func = getattr(module, function_name)
        return getattr(func, '__doc__', None)

    def get_class_methods(self, plugin_id: str, class_name: str) -> List[str]:
        """
        Get list of methods for a class.

        Args:
            plugin_id: ID of the loaded plugin
            class_name: Name of the class

        Returns:
            List of method names
        """
        cls = self.get_class(plugin_id, class_name)
        if not cls:
            return []

        return [method for method in dir(cls) if not method.startswith('_')]

    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload plugin module.

        Args:
            plugin_id: ID of the plugin to unload

        Returns:
            True if successfully unloaded
        """
        if plugin_id in self.loaded_modules:
            # Remove from loaded modules
            del self.loaded_modules[plugin_id]

            # Remove from sys.modules if it exists there
            if plugin_id in sys.modules:
                del sys.modules[plugin_id]

        if plugin_id in self.function_signatures:
            del self.function_signatures[plugin_id]

        return True

    def list_loaded_plugins(self) -> List[str]:
        """Get list of currently loaded plugin IDs."""
        return list(self.loaded_modules.keys())

    def reload_plugin(self, plugin_id: str) -> bool:
        """
        Reload a plugin module.

        Args:
            plugin_id: ID of the plugin to reload

        Returns:
            True if successfully reloaded
        """
        if plugin_id not in self.loaded_modules:
            return False

        # Get original plugin definition (this would need to be stored)
        # For now, just unload - full reload would need plugin definition
        return self.unload_plugin(plugin_id)


def detect_pybind11_module(path: str) -> bool:
    """
    Detect if file is a pybind11 Python extension module.

    Args:
        path: Path to the file

    Returns:
        True if it's a recognized pybind11 extension module
    """
    if not os.path.exists(path):
        return False

    path_obj = Path(path)

    # Check for Python extension module suffixes
    extensions = {'.so', '.pyd', '.dll'}  # .so on Linux/macOS, .pyd/.dll on Windows

    if path_obj.suffix.lower() not in extensions:
        return False

    # Additional checks could be added here to verify pybind11 signatures
    # For example, checking for specific symbols or metadata

    return True


def create_pybind11_plugin_definition(
    plugin_id: str,
    module_path: str,
    functions: List[str] = None,
    classes: Dict[str, List[str]] = None
) -> PluginDefinition:
    """
    Create plugin definition for pybind11 extension module.

    Args:
        plugin_id: Unique identifier for the plugin
        module_path: Path to the extension module file
        functions: List of function names exposed by the module
        classes: Dictionary mapping class names to their method names

    Returns:
        PluginDefinition configured for pybind11 plugin
    """
    signatures = {}
    if functions:
        signatures['functions'] = functions
    if classes:
        signatures['classes'] = classes

    return PluginDefinition(
        plugin_id=plugin_id,
        plugin_type="pybind11_module",
        binary_path=module_path,
        function_signatures=signatures,
        metadata={
            'language': 'C++',
            'binding_type': 'pybind11',
            'python_extension': True,
        }
    )