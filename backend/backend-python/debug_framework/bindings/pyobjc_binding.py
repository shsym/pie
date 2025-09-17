"""
Objective-C/Metal Plugin Bindings using PyObjC.

Provides interface for loading and interfacing with Objective-C/Metal libraries
as debug framework plugins on macOS.
"""

import os
import platform
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from ..models.plugin_definition import PluginDefinition

# PyObjC imports - only available on macOS
try:
    import objc
    from Foundation import NSBundle, NSString
    import Metal
    PYOBJC_AVAILABLE = True
except ImportError:
    PYOBJC_AVAILABLE = False


class PyObjCBinding:
    """
    Objective-C/Metal plugin binding using PyObjC for macOS Metal integration.
    """

    def __init__(self):
        if not PYOBJC_AVAILABLE:
            raise ImportError("PyObjC not available - macOS required for Metal bindings")

        if platform.system() != "Darwin":
            raise RuntimeError("PyObjC bindings only supported on macOS")

        self.loaded_bundles: Dict[str, NSBundle] = {}
        self.metal_libraries: Dict[str, Any] = {}
        self.function_signatures: Dict[str, Dict[str, Any]] = {}

    def load_plugin(self, plugin_def: PluginDefinition) -> bool:
        """
        Load Objective-C/Metal bundle or framework plugin.

        Args:
            plugin_def: Plugin definition with bundle path and signatures

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            bundle_path = plugin_def.binary_path
            if not bundle_path or not os.path.exists(bundle_path):
                return False

            # Load the bundle/framework
            bundle = NSBundle.bundleWithPath_(bundle_path)
            if not bundle:
                return False

            # Load the bundle into memory
            if not bundle.load():
                return False

            self.loaded_bundles[plugin_def.plugin_id] = bundle
            self.function_signatures[plugin_def.plugin_id] = plugin_def.function_signatures or {}

            # If this is a Metal library, handle it specially
            if plugin_def.plugin_type == "metal_library":
                self._load_metal_library(plugin_def)

            return True

        except Exception as e:
            print(f"Failed to load Objective-C plugin {plugin_def.plugin_id}: {e}")
            return False

    def _load_metal_library(self, plugin_def: PluginDefinition) -> None:
        """Load Metal shader library."""
        try:
            # Get default Metal device
            device = Metal.MTLCreateSystemDefaultDevice()
            if not device:
                return

            # Load Metal library from file
            library_path = plugin_def.binary_path
            with open(library_path, 'rb') as f:
                library_data = f.read()

            library = device.newLibraryWithData_error_(library_data, None)[0]
            if library:
                self.metal_libraries[plugin_def.plugin_id] = {
                    'library': library,
                    'device': device
                }

        except Exception as e:
            print(f"Failed to load Metal library {plugin_def.plugin_id}: {e}")

    def call_objective_c_method(
        self,
        plugin_id: str,
        class_name: str,
        method_name: str,
        *args
    ) -> Any:
        """
        Call Objective-C method from loaded bundle.

        Args:
            plugin_id: ID of the loaded plugin
            class_name: Name of the Objective-C class
            method_name: Name of the method to call
            *args: Arguments to pass to the method

        Returns:
            Method return value

        Raises:
            ValueError: If plugin, class, or method not found
        """
        if plugin_id not in self.loaded_bundles:
            raise ValueError(f"Plugin {plugin_id} not loaded")

        bundle = self.loaded_bundles[plugin_id]

        # Get the class
        cls = objc.lookUpClass(class_name)
        if not cls:
            raise ValueError(f"Class {class_name} not found in plugin {plugin_id}")

        # Create instance and call method
        instance = cls.alloc().init()
        if not hasattr(instance, method_name):
            raise ValueError(f"Method {method_name} not found in class {class_name}")

        method = getattr(instance, method_name)
        return method(*args)

    def get_metal_function(self, plugin_id: str, function_name: str) -> Optional[Any]:
        """
        Get Metal compute function from loaded library.

        Args:
            plugin_id: ID of the loaded Metal plugin
            function_name: Name of the Metal kernel function

        Returns:
            Metal function object or None
        """
        if plugin_id not in self.metal_libraries:
            return None

        library_info = self.metal_libraries[plugin_id]
        library = library_info['library']

        return library.newFunctionWithName_(function_name)

    def create_metal_compute_pipeline(
        self,
        plugin_id: str,
        function_name: str
    ) -> Optional[Any]:
        """
        Create Metal compute pipeline state for a kernel function.

        Args:
            plugin_id: ID of the loaded Metal plugin
            function_name: Name of the Metal kernel function

        Returns:
            Metal compute pipeline state or None
        """
        if plugin_id not in self.metal_libraries:
            return None

        library_info = self.metal_libraries[plugin_id]
        device = library_info['device']
        library = library_info['library']

        # Get the compute function
        function = library.newFunctionWithName_(function_name)
        if not function:
            return None

        # Create compute pipeline state
        try:
            pipeline_state = device.newComputePipelineStateWithFunction_error_(function, None)[0]
            return pipeline_state
        except Exception:
            return None

    def get_bundle_resource(self, plugin_id: str, resource_name: str, resource_type: str) -> Optional[str]:
        """
        Get path to bundle resource.

        Args:
            plugin_id: ID of the loaded plugin
            resource_name: Name of the resource
            resource_type: File extension of the resource

        Returns:
            Path to resource or None
        """
        if plugin_id not in self.loaded_bundles:
            return None

        bundle = self.loaded_bundles[plugin_id]
        return bundle.pathForResource_ofType_(resource_name, resource_type)

    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload plugin bundle.

        Args:
            plugin_id: ID of the plugin to unload

        Returns:
            True if successfully unloaded
        """
        if plugin_id in self.loaded_bundles:
            bundle = self.loaded_bundles[plugin_id]
            bundle.unload()
            del self.loaded_bundles[plugin_id]

        if plugin_id in self.metal_libraries:
            del self.metal_libraries[plugin_id]

        if plugin_id in self.function_signatures:
            del self.function_signatures[plugin_id]

        return True

    def list_loaded_plugins(self) -> List[str]:
        """Get list of currently loaded plugin IDs."""
        return list(self.loaded_bundles.keys())

    def get_metal_library_function_names(self, plugin_id: str) -> List[str]:
        """Get list of function names in Metal library."""
        if plugin_id not in self.metal_libraries:
            return []

        library = self.metal_libraries[plugin_id]['library']
        function_names = library.functionNames()

        return [str(name) for name in function_names]


def detect_objc_bundle(path: str) -> bool:
    """
    Detect if path is an Objective-C bundle or framework.

    Args:
        path: Path to the bundle/framework

    Returns:
        True if it's a recognized Objective-C bundle
    """
    if not os.path.exists(path):
        return False

    path_obj = Path(path)

    # Check for bundle/framework extensions
    if path_obj.suffix in {'.bundle', '.framework', '.app'}:
        return True

    # Check for Metal library
    if path_obj.suffix in {'.metallib', '.air'}:
        return True

    return False


def detect_metal_library(path: str) -> bool:
    """
    Detect if file is a Metal library.

    Args:
        path: Path to the file

    Returns:
        True if it's a Metal library
    """
    if not os.path.exists(path):
        return False

    path_obj = Path(path)
    return path_obj.suffix.lower() in {'.metallib', '.air'}


def create_objc_plugin_definition(
    plugin_id: str,
    bundle_path: str,
    classes: Dict[str, List[str]],
    plugin_type: str = "objc_bundle"
) -> PluginDefinition:
    """
    Create plugin definition for Objective-C bundle.

    Args:
        plugin_id: Unique identifier for the plugin
        bundle_path: Path to the bundle
        classes: Dictionary mapping class names to their method names
        plugin_type: Type of plugin (objc_bundle, metal_library)

    Returns:
        PluginDefinition configured for Objective-C plugin
    """
    return PluginDefinition(
        plugin_id=plugin_id,
        plugin_type=plugin_type,
        binary_path=bundle_path,
        function_signatures={'classes': classes},
        metadata={
            'language': 'Objective-C',
            'binding_type': 'pyobjc',
            'platform': 'macOS',
            'requires_metal': plugin_type == 'metal_library',
        }
    )


def create_metal_plugin_definition(
    plugin_id: str,
    library_path: str,
    kernel_functions: List[str]
) -> PluginDefinition:
    """
    Create plugin definition for Metal library.

    Args:
        plugin_id: Unique identifier for the plugin
        library_path: Path to the Metal library
        kernel_functions: List of kernel function names

    Returns:
        PluginDefinition configured for Metal plugin
    """
    return PluginDefinition(
        plugin_id=plugin_id,
        plugin_type="metal_library",
        binary_path=library_path,
        function_signatures={'kernels': kernel_functions},
        metadata={
            'language': 'Metal',
            'binding_type': 'pyobjc',
            'platform': 'macOS',
            'requires_metal': True,
        }
    )