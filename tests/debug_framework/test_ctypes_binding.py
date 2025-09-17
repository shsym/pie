"""
Tests for C/C++ plugin bindings using ctypes.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from debug_framework.bindings.ctypes_binding import (
    CTypesBinding,
    detect_c_library,
    create_c_plugin_definition
)
from debug_framework.models.plugin_definition import PluginDefinition


class TestCTypesBinding:
    """Test cases for CTypesBinding class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.binding = CTypesBinding()

    def test_init(self):
        """Test initialization of CTypesBinding."""
        assert self.binding.loaded_libraries == {}
        assert self.binding.function_signatures == {}

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_load_plugin_success(self, mock_exists, mock_cdll):
        """Test successful plugin loading."""
        mock_exists.return_value = True
        mock_lib = Mock()
        mock_cdll.return_value = mock_lib

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/path/to/lib.so",
            function_signatures={
                'test_func': {
                    'argtypes': ['int', 'float'],
                    'restype': 'int'
                }
            }
        )

        result = self.binding.load_plugin(plugin_def)

        assert result is True
        assert "test_plugin" in self.binding.loaded_libraries
        assert self.binding.loaded_libraries["test_plugin"] == mock_lib

    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_load_plugin_file_not_found(self, mock_exists):
        """Test plugin loading when file doesn't exist."""
        mock_exists.return_value = False

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/nonexistent/lib.so"
        )

        result = self.binding.load_plugin(plugin_def)
        assert result is False

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_load_plugin_cdll_exception(self, mock_exists, mock_cdll):
        """Test plugin loading when CDLL raises exception."""
        mock_exists.return_value = True
        mock_cdll.side_effect = OSError("Failed to load library")

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/path/to/lib.so"
        )

        result = self.binding.load_plugin(plugin_def)
        assert result is False

    def test_convert_to_ctypes(self):
        """Test type conversion to ctypes."""
        import ctypes

        type_list = ['int', 'float', 'char*', 'void*']
        result = self.binding._convert_to_ctypes(type_list)

        expected = [ctypes.c_int, ctypes.c_float, ctypes.c_char_p, ctypes.c_void_p]
        assert result == expected

    def test_convert_to_ctype_single(self):
        """Test single type conversion."""
        import ctypes

        assert self.binding._convert_to_ctype_single('int') == ctypes.c_int
        assert self.binding._convert_to_ctype_single('float') == ctypes.c_float
        assert self.binding._convert_to_ctype_single('void') is None

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_call_function_success(self, mock_exists, mock_cdll):
        """Test successful function call."""
        mock_exists.return_value = True
        mock_lib = Mock()
        mock_func = Mock(return_value=42)
        mock_lib.test_func = mock_func
        mock_cdll.return_value = mock_lib

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/path/to/lib.so"
        )

        self.binding.load_plugin(plugin_def)
        result = self.binding.call_function("test_plugin", "test_func", 1, 2.0)

        assert result == 42
        mock_func.assert_called_once_with(1, 2.0)

    def test_call_function_plugin_not_loaded(self):
        """Test calling function on unloaded plugin."""
        with pytest.raises(ValueError, match="Plugin .* not loaded"):
            self.binding.call_function("nonexistent", "test_func")

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_call_function_not_found(self, mock_exists, mock_cdll):
        """Test calling non-existent function."""
        mock_exists.return_value = True
        mock_lib = Mock()
        del mock_lib.nonexistent_func  # Ensure attribute doesn't exist
        mock_cdll.return_value = mock_lib

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/path/to/lib.so"
        )

        self.binding.load_plugin(plugin_def)

        with pytest.raises(ValueError, match="Function .* not found"):
            self.binding.call_function("test_plugin", "nonexistent_func")

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_unload_plugin(self, mock_exists, mock_cdll):
        """Test plugin unloading."""
        mock_exists.return_value = True
        mock_cdll.return_value = Mock()

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/path/to/lib.so"
        )

        self.binding.load_plugin(plugin_def)
        result = self.binding.unload_plugin("test_plugin")

        assert result is True
        assert "test_plugin" not in self.binding.loaded_libraries

    def test_unload_nonexistent_plugin(self):
        """Test unloading non-existent plugin."""
        result = self.binding.unload_plugin("nonexistent")
        assert result is False

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_list_loaded_plugins(self, mock_exists, mock_cdll):
        """Test listing loaded plugins."""
        mock_exists.return_value = True
        mock_cdll.return_value = Mock()

        plugin_def1 = PluginDefinition(
            plugin_id="plugin1",
            plugin_type="c_library",
            binary_path="/path/to/lib1.so"
        )

        plugin_def2 = PluginDefinition(
            plugin_id="plugin2",
            plugin_type="c_library",
            binary_path="/path/to/lib2.so"
        )

        self.binding.load_plugin(plugin_def1)
        self.binding.load_plugin(plugin_def2)

        plugins = self.binding.list_loaded_plugins()
        assert set(plugins) == {"plugin1", "plugin2"}

    @patch('debug_framework.bindings.ctypes_binding.ctypes.CDLL')
    @patch('debug_framework.bindings.ctypes_binding.os.path.exists')
    def test_get_plugin_functions(self, mock_exists, mock_cdll):
        """Test getting plugin functions."""
        mock_exists.return_value = True
        mock_cdll.return_value = Mock()

        plugin_def = PluginDefinition(
            plugin_id="test_plugin",
            plugin_type="c_library",
            binary_path="/path/to/lib.so",
            function_signatures={
                'func1': {'argtypes': ['int']},
                'func2': {'argtypes': ['float']}
            }
        )

        self.binding.load_plugin(plugin_def)
        functions = self.binding.get_plugin_functions("test_plugin")

        assert set(functions) == {"func1", "func2"}

    def test_get_plugin_functions_not_loaded(self):
        """Test getting functions for unloaded plugin."""
        functions = self.binding.get_plugin_functions("nonexistent")
        assert functions == []


class TestDetectCLibrary:
    """Test cases for C library detection."""

    def test_detect_c_library_valid_extensions(self):
        """Test detection with valid C library extensions."""
        with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as f:
            temp_path = f.name

        try:
            assert detect_c_library(temp_path) is True
        finally:
            os.unlink(temp_path)

        # Test other extensions without creating files
        assert detect_c_library('/path/to/lib.dll') is False  # File doesn't exist
        assert detect_c_library('/path/to/lib.dylib') is False  # File doesn't exist
        assert detect_c_library('/path/to/lib.a') is False  # File doesn't exist

    def test_detect_c_library_invalid_extension(self):
        """Test detection with invalid extensions."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            assert detect_c_library(temp_path) is False
        finally:
            os.unlink(temp_path)

    def test_detect_c_library_nonexistent_file(self):
        """Test detection with non-existent file."""
        assert detect_c_library('/nonexistent/path/lib.so') is False


class TestCreateCPluginDefinition:
    """Test cases for creating C plugin definitions."""

    def test_create_c_plugin_definition(self):
        """Test creating C plugin definition."""
        functions = {
            'add': {
                'argtypes': ['int', 'int'],
                'restype': 'int'
            },
            'multiply': {
                'argtypes': ['float', 'float'],
                'restype': 'float'
            }
        }

        plugin_def = create_c_plugin_definition(
            plugin_id="math_lib",
            library_path="/path/to/math.so",
            functions=functions
        )

        assert plugin_def.plugin_id == "math_lib"
        assert plugin_def.plugin_type == "c_library"
        assert plugin_def.binary_path == "/path/to/math.so"
        assert plugin_def.function_signatures == functions
        assert plugin_def.metadata['language'] == 'C/C++'
        assert plugin_def.metadata['binding_type'] == 'ctypes'
        assert 'platform' in plugin_def.metadata