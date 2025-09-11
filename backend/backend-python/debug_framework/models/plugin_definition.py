"""
PluginDefinition model for the debug framework.

This module defines the PluginDefinition data model which manages configuration
and metadata for alternative backend implementations that can be plugged into
the validation framework.
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from debug_framework.services import database_manager


class PluginDefinition:
    """
    Configuration and metadata for alternative backend implementations.

    Manages plugin source files, compilation configuration, interface mapping,
    and validation rules for backend implementations that can be dynamically
    compiled and loaded into the validation framework.
    """

    # Valid target platforms for plugin compilation
    VALID_TARGET_PLATFORMS = {"metal", "cpp", "objc", "cuda"}

    # Valid checkpoint names for interface mapping
    VALID_CHECKPOINT_NAMES = {
        "post_embedding", "post_rope", "post_attention",
        "pre_mlp", "post_mlp", "final_output"
    }

    # Required fields in compile configuration
    REQUIRED_COMPILE_CONFIG_FIELDS = {"target_platform"}

    def __init__(
        self,
        name: str,
        version: str,
        target_platform: str,
        source_files: List[str],
        compile_config: Dict[str, Any],
        interface_config: Dict[str, Any],
        id: Optional[int] = None,
        is_compiled: bool = False,
        binary_path: Optional[str] = None,
        compilation_errors: Optional[str] = None,
        last_compiled_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a PluginDefinition instance.

        Args:
            name: Plugin name following naming convention (alphanumeric and underscores only)
            version: Semantic version string (e.g., "1.0.0", "2.1.0-alpha")
            target_platform: Target platform for compilation (metal, cpp, objc, cuda)
            source_files: List of source file paths that must exist
            compile_config: Compilation configuration including target platform requirements
            interface_config: Interface mapping and validation configuration
            id: Unique plugin definition identifier (auto-assigned if None)
            is_compiled: Whether plugin has been successfully compiled
            binary_path: Path to compiled binary (set after successful compilation)
            compilation_errors: Error messages from last compilation attempt
            last_compiled_at: Timestamp of last successful compilation
            created_at: Plugin definition creation timestamp (defaults to now)

        Raises:
            ValueError: If validation rules are violated
        """
        # Validate name
        self._validate_name(name)

        # Validate version
        self._validate_version(version)

        # Validate target platform
        self._validate_target_platform(target_platform)

        # Validate source files
        self._validate_source_files(source_files)

        # Validate compile configuration
        self._validate_compile_config(compile_config)

        # Validate interface configuration
        self._validate_interface_config(interface_config)

        # Store attributes
        self.id = id  # None until saved to database
        self.name = name
        self.version = version
        self.target_platform = target_platform
        self.source_files = source_files.copy()  # Deep copy for safety
        self.compile_config = compile_config.copy()  # Deep copy for safety
        self.interface_config = interface_config.copy()  # Deep copy for safety
        self.is_compiled = is_compiled
        self.binary_path = binary_path
        self.compilation_errors = compilation_errors
        self.last_compiled_at = last_compiled_at
        self.created_at = created_at or datetime.now()

        # Interface validator instance (lazy-loaded)
        self.interface_validator = None

    def _validate_name(self, name: str) -> None:
        """Validate plugin name follows naming conventions."""
        if not name:
            raise ValueError("name is required and cannot be empty")

        # Plugin names must be alphanumeric with underscores, no hyphens or spaces
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
            raise ValueError("name must follow naming convention: alphanumeric and underscores only, start with letter")

    def _validate_version(self, version: str) -> None:
        """Validate version follows semantic versioning."""
        if not version:
            raise ValueError("version is required and cannot be empty")

        # Semantic versioning pattern: MAJOR.MINOR.PATCH with optional pre-release
        semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?$'
        if not re.match(semver_pattern, version):
            raise ValueError("version must follow semantic versioning format (e.g., '1.0.0', '2.1.0-alpha')")

    def _validate_target_platform(self, target_platform: str) -> None:
        """Validate target platform is supported."""
        if not target_platform:
            raise ValueError("target_platform is required and cannot be empty")

        if target_platform not in self.VALID_TARGET_PLATFORMS:
            valid_platforms = ", ".join(sorted(self.VALID_TARGET_PLATFORMS))
            raise ValueError(f"Invalid target platform '{target_platform}'. Must be one of: {valid_platforms}")

    def _validate_source_files(self, source_files: List[str]) -> None:
        """Validate source files exist and are accessible."""
        if not source_files:
            raise ValueError("source_files must contain at least one valid file path")

        if not isinstance(source_files, list):
            raise ValueError("source_files must be a list of file paths")

        # Check that all source files exist (both relative and absolute paths)
        for source_file in source_files:
            if not isinstance(source_file, str):
                raise ValueError("All source files must be valid file path strings")

            if not os.path.exists(source_file):
                raise ValueError(f"source file does not exist: {source_file}")

    def _validate_compile_config(self, compile_config: Dict[str, Any]) -> None:
        """Validate compile configuration contains required fields."""
        if not isinstance(compile_config, dict):
            raise ValueError("compile_config must be a dictionary")

        # Check for required fields
        missing_fields = self.REQUIRED_COMPILE_CONFIG_FIELDS - set(compile_config.keys())
        if missing_fields:
            raise ValueError("compile_config must specify target platform requirements")

    def _validate_interface_config(self, interface_config: Dict[str, Any]) -> None:
        """Validate interface configuration contains valid checkpoint mappings."""
        if not isinstance(interface_config, dict):
            raise ValueError("interface_config must be a dictionary")

        # If checkpoint_mapping exists, validate checkpoint names
        checkpoint_mapping = interface_config.get("checkpoint_mapping", {})
        if checkpoint_mapping:
            invalid_checkpoints = set(checkpoint_mapping.keys()) - self.VALID_CHECKPOINT_NAMES
            if invalid_checkpoints:
                valid_names = ", ".join(sorted(self.VALID_CHECKPOINT_NAMES))
                raise ValueError(f"interface_config must map to valid checkpoint names. Valid names: {valid_names}")

    def mark_compiled(self, binary_path: str) -> None:
        """
        Mark plugin as successfully compiled.

        Args:
            binary_path: Path to the compiled binary

        Raises:
            ValueError: If binary_path is empty or invalid
        """
        if not binary_path:
            raise ValueError("binary_path is required and cannot be empty")

        self.is_compiled = True
        self.binary_path = binary_path
        self.compilation_errors = None
        self.last_compiled_at = datetime.now()

    def mark_compilation_failed(self, error_message: str) -> None:
        """
        Mark plugin compilation as failed.

        Args:
            error_message: Description of compilation failure
        """
        self.is_compiled = False
        self.compilation_errors = error_message
        # Keep binary_path as is (don't clear it) for potential debugging
        # last_compiled_at is not updated on failure

    def create_interface_validator(self):
        """
        Create an InterfaceValidator for this plugin.

        Returns:
            InterfaceValidator instance configured for this plugin

        Note:
            This method imports InterfaceValidator dynamically to avoid circular imports.
        """
        from debug_framework.models.interface_validator import InterfaceValidator
        return InterfaceValidator(plugin_id=self.id)

    def get_interface_validator(self):
        """
        Get the interface validator instance for this plugin.

        Returns:
            InterfaceValidator instance, or None if not set
        """
        return self.interface_validator

    def compile(self) -> Dict[str, Any]:
        """
        Trigger compilation of this plugin using the CompilationEngine.

        Returns:
            Dictionary with compilation results:
            - success: bool indicating if compilation succeeded
            - binary_path: str path to compiled binary if successful
            - errors: str error message if compilation failed

        Note:
            This method imports CompilationEngine dynamically to avoid circular imports.
        """
        from debug_framework.services.compilation_engine import CompilationEngine

        compilation_engine = CompilationEngine()
        result = compilation_engine.compile_plugin(self)

        if result["success"]:
            self.mark_compiled(result["binary_path"])
        else:
            self.mark_compilation_failed(result.get("errors", "Unknown compilation error"))

        return result

    def save(self) -> int:
        """
        Save plugin definition to database.

        Returns:
            Database ID of the saved plugin definition

        Raises:
            ValueError: If plugin name already exists in database
        """
        db_manager = database_manager.DatabaseManager()

        # Prepare data for database insertion
        plugin_data = {
            "name": self.name,
            "version": self.version,
            "target_platform": self.target_platform,
            "source_files": json.dumps(self.source_files),
            "compile_config": json.dumps(self.compile_config),
            "interface_config": json.dumps(self.interface_config),
            "is_compiled": self.is_compiled,
            "binary_path": self.binary_path,
            "compilation_errors": self.compilation_errors,
            "last_compiled_at": self.last_compiled_at.isoformat() if self.last_compiled_at else None,
            "created_at": self.created_at.isoformat()
        }

        try:
            if self.id is None:
                # Insert new plugin definition
                self.id = db_manager.insert_plugin_definition(plugin_data)
            else:
                # Update existing plugin definition
                db_manager.update_plugin_definition(self.id, plugin_data)
        except Exception as e:
            if "UNIQUE constraint failed" in str(e) or "name" in str(e).lower():
                raise ValueError(f"Plugin name '{self.name}' already exists")
            raise

        return self.id

    @classmethod
    def load(cls, plugin_id: int) -> 'PluginDefinition':
        """
        Load plugin definition from database by ID.

        Args:
            plugin_id: Database ID of the plugin definition

        Returns:
            PluginDefinition instance loaded from database

        Raises:
            ValueError: If plugin not found in database
        """
        db_manager = database_manager.DatabaseManager()
        plugin_data = db_manager.get_plugin_definition(plugin_id)

        if not plugin_data:
            raise ValueError(f"Plugin definition with ID {plugin_id} not found")

        return cls.from_dict(plugin_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert PluginDefinition to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "target_platform": self.target_platform,
            "source_files": self.source_files,
            "compile_config": self.compile_config,
            "interface_config": self.interface_config,
            "is_compiled": self.is_compiled,
            "binary_path": self.binary_path,
            "compilation_errors": self.compilation_errors,
            "last_compiled_at": self.last_compiled_at.isoformat() if self.last_compiled_at else None,
            "created_at": self.created_at.isoformat()
        }

    def to_json(self) -> str:
        """Convert PluginDefinition to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginDefinition':
        """Create PluginDefinition from dictionary representation."""
        # Parse JSON fields
        source_files = json.loads(data.get("source_files", "[]")) if isinstance(data.get("source_files"), str) else data.get("source_files", [])
        compile_config = json.loads(data.get("compile_config", "{}")) if isinstance(data.get("compile_config"), str) else data.get("compile_config", {})
        interface_config = json.loads(data.get("interface_config", "{}")) if isinstance(data.get("interface_config"), str) else data.get("interface_config", {})

        # Parse timestamp fields
        last_compiled_at = None
        if data.get("last_compiled_at"):
            if isinstance(data["last_compiled_at"], str):
                try:
                    last_compiled_at = datetime.fromisoformat(data["last_compiled_at"].replace('Z', '+00:00'))
                except ValueError:
                    pass  # Keep as None if parsing fails
            elif isinstance(data["last_compiled_at"], datetime):
                last_compiled_at = data["last_compiled_at"]

        created_at = None
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                try:
                    created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
                except ValueError:
                    created_at = datetime.now()  # Default to now if parsing fails
            elif isinstance(data["created_at"], datetime):
                created_at = data["created_at"]

        return cls(
            id=data.get("id"),
            name=data["name"],
            version=data["version"],
            target_platform=data["target_platform"],
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config,
            is_compiled=data.get("is_compiled", False),
            binary_path=data.get("binary_path"),
            compilation_errors=data.get("compilation_errors"),
            last_compiled_at=last_compiled_at,
            created_at=created_at
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'PluginDefinition':
        """Create PluginDefinition from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of PluginDefinition."""
        status = "compiled" if self.is_compiled else "not compiled"
        return f"PluginDefinition(name='{self.name}', version='{self.version}', platform='{self.target_platform}', status={status})"

    def __repr__(self) -> str:
        """Detailed representation of PluginDefinition."""
        return (f"PluginDefinition(id={self.id}, name='{self.name}', version='{self.version}', "
                f"target_platform='{self.target_platform}', is_compiled={self.is_compiled})")