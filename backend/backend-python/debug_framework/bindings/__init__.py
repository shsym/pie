"""
Debug Framework Bindings

Plugin interface bindings for ctypes (C), pybind11 (C++),
and PyObjC (Objective-C/Metal).
"""

from .ctypes_binding import CTypesBinding, detect_c_library, create_c_plugin_definition
from .pybind11_binding import Pybind11Binding, detect_pybind11_module, create_pybind11_plugin_definition

# PyObjC bindings only available on macOS
try:
    from .pyobjc_binding import (
        PyObjCBinding,
        detect_objc_bundle,
        detect_metal_library,
        create_objc_plugin_definition,
        create_metal_plugin_definition
    )
    METAL_SUPPORT = True
except ImportError:
    METAL_SUPPORT = False

from .signature_extractor import (
    extract_signatures,
    extract_c_signatures,
    extract_metal_signatures,
    extract_cuda_signatures
)

__all__ = [
    'CTypesBinding',
    'Pybind11Binding',
    'detect_c_library',
    'detect_pybind11_module',
    'create_c_plugin_definition',
    'create_pybind11_plugin_definition',
    'extract_signatures',
    'extract_c_signatures',
    'extract_metal_signatures',
    'extract_cuda_signatures',
    'METAL_SUPPORT'
]

# Add Metal exports if available
if METAL_SUPPORT:
    __all__.extend([
        'PyObjCBinding',
        'detect_objc_bundle',
        'detect_metal_library',
        'create_objc_plugin_definition',
        'create_metal_plugin_definition'
    ])