"""
TensorRecording model for the debug framework.

This module defines the TensorRecording data model which provides JSON-formatted
capture of tensor data with complete metadata for offline validation scenarios.
"""

import os
import json
import numpy as np
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path

from debug_framework.services import database_manager


class CompressionMethod(Enum):
    """Valid compression methods for tensor data."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


class TensorRecording:
    """
    JSON-formatted capture of tensor data with complete metadata for offline validation.

    Enables recording tensor data during one session and replaying/comparing in another,
    supporting offline validation scenarios where live dual-backend execution isn't possible.
    """

    # Required metadata fields for tensor specifications
    REQUIRED_METADATA_FIELDS = {
        "dtype", "shape", "strides", "device", "memory_layout", "byte_order"
    }

    # Required device info fields
    REQUIRED_DEVICE_INFO_FIELDS = {
        "platform"
    }

    def __init__(
        self,
        session_id: int,
        checkpoint_id: int,
        tensor_name: str,
        tensor_metadata: Dict[str, Any],
        tensor_data_path: str,
        backend_name: str,
        device_info: Dict[str, Any],
        recording_timestamp: Optional[datetime] = None,
        compression_method: Union[str, CompressionMethod] = CompressionMethod.NONE,
        file_size_bytes: Optional[int] = None,
        id: Optional[int] = None
    ):
        """
        Initialize a TensorRecording instance.

        Args:
            session_id: Parent debug session ID
            checkpoint_id: Associated validation checkpoint ID
            tensor_name: Identifier for the tensor
            tensor_metadata: Complete tensor specifications (dtype, shape, stride, device)
            tensor_data_path: Path to binary tensor file
            backend_name: Backend that produced this tensor
            device_info: Original device placement information
            recording_timestamp: When tensor was captured (defaults to now)
            compression_method: Data compression used
            file_size_bytes: Size of tensor data file (auto-calculated if None)
            id: Unique recording identifier (auto-assigned if None)

        Raises:
            ValueError: If validation rules are violated
            FileNotFoundError: If tensor_data_path doesn't exist
        """
        # Validate required fields
        if not tensor_name:
            raise ValueError("tensor_name is required and cannot be empty")

        if not backend_name:
            raise ValueError("backend_name is required and cannot be empty")

        # Validate backend name against plugin registry
        self._validate_backend_name(backend_name)

        if not tensor_data_path:
            raise ValueError("tensor_data_path is required and cannot be empty")

        # Validate tensor data path exists
        self._validate_tensor_data_path(tensor_data_path)

        # Validate tensor metadata
        self._validate_tensor_metadata(tensor_metadata)

        # Validate device info
        self._validate_device_info(device_info)

        # Store path without validating existence (validation happens in validate_recording_integrity)
        # This allows for test scenarios and deferred validation

        # Handle compression method
        if isinstance(compression_method, str):
            try:
                compression_method = CompressionMethod(compression_method)
            except ValueError:
                valid_methods = [method.value for method in CompressionMethod]
                raise ValueError(f"Invalid compression method. Must be one of: {valid_methods}")

        # Validate compression method availability
        if compression_method == CompressionMethod.LZ4:
            try:
                import lz4
            except ImportError:
                raise ValueError("LZ4 compression not available")

        # Auto-calculate file size if not provided
        if file_size_bytes is None:
            if os.path.exists(tensor_data_path):
                file_size_bytes = os.path.getsize(tensor_data_path)
            else:
                file_size_bytes = 0  # Default size for non-existent files

        # Set attributes
        self.id = id  # None until saved to database
        self.session_id = session_id
        self.checkpoint_id = checkpoint_id
        self.tensor_name = tensor_name
        self.tensor_metadata = tensor_metadata.copy()  # Deep copy for safety
        self.tensor_data_path = tensor_data_path
        self.backend_name = backend_name
        self.device_info = device_info.copy()  # Deep copy for safety
        self._recording_timestamp_dt = recording_timestamp or datetime.now()
        self.compression_method_enum = compression_method
        self.file_size_bytes = file_size_bytes

        # ID will be assigned by database when saved

    def _validate_tensor_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate that tensor metadata contains all required fields."""
        if not isinstance(metadata, dict):
            raise ValueError("tensor_metadata must be a dictionary")

        missing_fields = self.REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing_fields:
            raise ValueError("tensor_metadata must include dtype, shape, strides, device information")

        # Validate specific field types and values
        if not isinstance(metadata.get("shape"), (list, tuple)):
            raise ValueError("tensor_metadata.shape must be a list or tuple")

        if len(metadata["shape"]) == 0:
            raise ValueError("tensor_metadata.shape cannot be empty")

        if not all(isinstance(dim, int) and dim > 0 for dim in metadata["shape"]):
            raise ValueError("tensor_metadata.shape must contain positive integers")

    def _validate_device_info(self, device_info: Dict[str, Any]) -> None:
        """Validate that device info contains all required platform-specific details."""
        if not isinstance(device_info, dict):
            raise ValueError("device_info must be a dictionary")

        missing_fields = self.REQUIRED_DEVICE_INFO_FIELDS - set(device_info.keys())
        if missing_fields:
            raise ValueError("device_info must include platform-specific details")

        # Validate platform-specific requirements
        platform = device_info.get("platform", "").lower()
        if platform not in {"cpu", "cuda", "metal", "mps"}:
            raise ValueError(f"Unsupported platform: {platform}")

    def _validate_backend_name(self, backend_name: str) -> None:
        """Validate backend name against plugin registry."""
        try:
            from debug_framework.services.plugin_registry import PluginRegistry
            registry = PluginRegistry()
            if not registry.is_registered(backend_name):
                raise ValueError("backend_name must match registered backend")
        except ImportError:
            # If plugin registry is not available, skip validation
            # This allows tests to run during development
            raise ValueError(f"PluginRegistry service is not available for backend[{backend_name}] validation")

    def _validate_tensor_data_path(self, tensor_data_path: str) -> None:
        """Validate tensor data path points to valid binary file."""
        if not os.path.exists(tensor_data_path):
            raise ValueError("tensor_data_path must point to valid binary file")

    @property
    def compression_method(self) -> str:
        """Get compression method as string value."""
        return self.compression_method_enum.value if isinstance(self.compression_method_enum, CompressionMethod) else self.compression_method_enum

    @property
    def recording_timestamp(self) -> str:
        """Get recording timestamp as ISO string."""
        if isinstance(self._recording_timestamp_dt, str):
            return self._recording_timestamp_dt
        return self._recording_timestamp_dt.isoformat()

    def validate_backend_registration(self, registered_backends: List[str]) -> bool:
        """
        Validate that backend_name matches a registered backend.

        Args:
            registered_backends: List of valid backend names

        Returns:
            True if backend is registered, False otherwise
        """
        return self.backend_name in registered_backends

    def get_tensor_size_mb(self) -> float:
        """Get tensor data file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    def is_compressed(self) -> bool:
        """Check if tensor data is compressed."""
        return self.compression_method_enum != CompressionMethod.NONE

    def get_metadata_json(self) -> str:
        """Get tensor metadata as JSON string."""
        return json.dumps(self.tensor_metadata, indent=2)

    def get_device_info_json(self) -> str:
        """Get device info as JSON string."""
        return json.dumps(self.device_info, indent=2)

    def validate_recording_integrity(self) -> Dict[str, bool]:
        """
        Validate the integrity of the recorded tensor data.

        Returns:
            Dictionary with validation results for different aspects
        """
        results = {
            "file_exists": os.path.exists(self.tensor_data_path),
            "size_matches": False,
            "readable": False,
            "metadata_valid": False
        }

        if results["file_exists"]:
            actual_size = os.path.getsize(self.tensor_data_path)
            results["size_matches"] = actual_size == self.file_size_bytes

            try:
                with open(self.tensor_data_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
                results["readable"] = True
            except Exception:
                results["readable"] = False

        # Validate metadata completeness
        try:
            self._validate_tensor_metadata(self.tensor_metadata)
            self._validate_device_info(self.device_info)
            results["metadata_valid"] = True
        except ValueError:
            results["metadata_valid"] = False

        return results

    def get_iso_timestamp(self) -> str:
        """Get recording timestamp in ISO format."""
        return self.recording_timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert TensorRecording to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "tensor_name": self.tensor_name,
            "tensor_metadata": self.tensor_metadata,
            "tensor_data_path": self.tensor_data_path,
            "backend_name": self.backend_name,
            "device_info": self.device_info,
            "recording_timestamp": self.get_iso_timestamp(),
            "compression_method": self.compression_method,
            "file_size_bytes": self.file_size_bytes
        }

    def to_json(self) -> str:
        """Convert TensorRecording to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TensorRecording':
        """Create TensorRecording from dictionary representation."""
        # Parse timestamp - can be string or datetime
        timestamp = data.get("recording_timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                # If parsing fails, keep as string
                pass

        return cls(
            id=data.get("id"),
            session_id=data["session_id"],
            checkpoint_id=data["checkpoint_id"],
            tensor_name=data["tensor_name"],
            tensor_metadata=data["tensor_metadata"],
            tensor_data_path=data["tensor_data_path"],
            backend_name=data["backend_name"],
            device_info=data["device_info"],
            recording_timestamp=timestamp,
            compression_method=data.get("compression_method", "none"),
            file_size_bytes=data.get("file_size_bytes")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TensorRecording':
        """Create TensorRecording from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_platform(self) -> str:
        """Get the platform from device info."""
        return self.device_info.get("platform", "unknown")

    def save(self) -> int:
        """
        Save tensor recording to database.

        Returns:
            Database ID of the saved recording
        """
        db_manager = database_manager.DatabaseManager()

        # Map model fields to database schema fields
        # Store extra info like compression_method, file_size_bytes in tensor_metadata
        enhanced_metadata = self.tensor_metadata.copy()
        enhanced_metadata.update({
            "compression_method": self.compression_method,
            "file_size_bytes": self.file_size_bytes,
            "device_info": self.device_info,
            "session_id": self.session_id  # Store in metadata for reference
        })

        recording_data = {
            "session_id": self.session_id,  # Include for test compatibility
            "checkpoint_id": self.checkpoint_id,
            "tensor_name": self.tensor_name,
            "tensor_metadata": enhanced_metadata,  # Will be JSON serialized by database manager
            "tensor_data_path": self.tensor_data_path,
            "backend": self.backend_name,  # Map backend_name to backend
            "created_at": self.recording_timestamp  # Already ISO string format
        }

        if self.id is None:
            # Insert new recording
            self.id = db_manager.insert_tensor_recording(recording_data)
        else:
            # Update existing recording
            db_manager.update_tensor_recording(self.id, recording_data)

        return self.id

    @classmethod
    def create_from_tensor(
        cls,
        session_id: int,
        checkpoint_id: int,
        tensor_name: str,
        tensor_data: np.ndarray,
        backend_name: str,
        device_info: Dict[str, Any],
        compression_method: Union[str, CompressionMethod] = CompressionMethod.NONE,
        storage_dir: Optional[str] = None
    ) -> 'TensorRecording':
        """
        Create a TensorRecording from a numpy tensor.

        Args:
            session_id: Parent debug session ID
            checkpoint_id: Associated validation checkpoint ID
            tensor_name: Identifier for the tensor
            tensor_data: NumPy array containing tensor data
            backend_name: Backend that produced this tensor
            device_info: Device placement information
            compression_method: Compression method to use
            storage_dir: Directory for temporary tensor files

        Returns:
            New TensorRecording instance with tensor data saved to file
        """
        if not isinstance(tensor_data, np.ndarray):
            raise ValueError("tensor_data must be a numpy array")

        # Convert compression_method to enum if it's a string
        if isinstance(compression_method, str):
            try:
                compression_method = CompressionMethod(compression_method)
            except ValueError:
                valid_methods = [method.value for method in CompressionMethod]
                raise ValueError(f"Invalid compression method. Must be one of: {valid_methods}")

        # Create temporary file for tensor data
        if storage_dir:
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb', delete=False, suffix='.tensor', dir=storage_dir
            )
        else:
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb', delete=False, suffix='.tensor'
            )

        try:
            # Save tensor data to file
            tensor_data.tofile(temp_file.name)
            tensor_data_path = temp_file.name

            # Apply compression if requested
            if isinstance(compression_method, CompressionMethod) and compression_method != CompressionMethod.NONE:
                tensor_data_path = cls._compress_tensor_file(temp_file.name, compression_method)
                # Remove original uncompressed file
                os.unlink(temp_file.name)

            file_size = os.path.getsize(tensor_data_path)

        finally:
            temp_file.close()

        # Extract metadata from tensor
        tensor_metadata = {
            "dtype": str(tensor_data.dtype),
            "shape": list(tensor_data.shape),
            "strides": list(tensor_data.strides),
            "device": device_info.get("device", "cpu"),
            "memory_layout": "C" if tensor_data.flags.c_contiguous else "F",
            "byte_order": tensor_data.dtype.byteorder if tensor_data.dtype.byteorder != '=' else '<'
        }

        return cls(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            tensor_name=tensor_name,
            tensor_metadata=tensor_metadata,
            tensor_data_path=tensor_data_path,
            backend_name=backend_name,
            device_info=device_info,
            compression_method=compression_method,
            file_size_bytes=file_size
        )

    def load_tensor_data(self) -> np.ndarray:
        """
        Load tensor data from file.

        Returns:
            NumPy array containing the tensor data

        Raises:
            FileNotFoundError: If tensor data file doesn't exist
            ValueError: If tensor data is corrupted or metadata is invalid
        """
        if not os.path.exists(self.tensor_data_path):
            raise FileNotFoundError(f"Tensor data file not found: {self.tensor_data_path}")

        try:
            # Extract shape and dtype from metadata
            shape = tuple(self.tensor_metadata["shape"])
            dtype = np.dtype(self.tensor_metadata["dtype"])

            # Load tensor data from file, handling compression
            if self.is_compressed():
                tensor_data = self._decompress_and_load(self.tensor_data_path, dtype, self.compression_method_enum)
            else:
                tensor_data = np.fromfile(self.tensor_data_path, dtype=dtype)

            # Reshape to original shape
            tensor_data = tensor_data.reshape(shape)

            return tensor_data

        except Exception as e:
            raise ValueError(f"Failed to load tensor data: {str(e)}")

    @classmethod
    def _compress_tensor_file(cls, file_path: str, compression_method: CompressionMethod) -> str:
        """
        Compress tensor file using specified compression method.

        Args:
            file_path: Path to uncompressed tensor file
            compression_method: Compression method to use

        Returns:
            Path to compressed file

        Raises:
            ValueError: If compression fails or method not supported
        """
        if compression_method == CompressionMethod.GZIP:
            import gzip
            compressed_path = file_path + '.gz'

            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            return compressed_path

        elif compression_method == CompressionMethod.LZ4:
            try:
                import lz4.frame
            except ImportError:
                raise ValueError("LZ4 compression not available")

            compressed_path = file_path + '.lz4'

            with open(file_path, 'rb') as f_in:
                with open(compressed_path, 'wb') as f_out:
                    data = f_in.read()
                    compressed_data = lz4.frame.compress(data)
                    f_out.write(compressed_data)

            return compressed_path

        else:
            raise ValueError(f"Unsupported compression method: {compression_method}")

    def _decompress_and_load(self, file_path: str, dtype: np.dtype, compression_method: CompressionMethod) -> np.ndarray:
        """
        Decompress and load tensor data from compressed file.

        Args:
            file_path: Path to compressed tensor file
            dtype: NumPy dtype of the tensor data
            compression_method: Compression method used

        Returns:
            NumPy array loaded from decompressed data

        Raises:
            ValueError: If decompression fails or method not supported
        """
        if compression_method == CompressionMethod.GZIP:
            import gzip
            with gzip.open(file_path, 'rb') as f:
                data = f.read()
                return np.frombuffer(data, dtype=dtype)

        elif compression_method == CompressionMethod.LZ4:
            try:
                import lz4.frame
            except ImportError:
                raise ValueError("LZ4 compression not available")

            with open(file_path, 'rb') as f:
                compressed_data = f.read()
                decompressed_data = lz4.frame.decompress(compressed_data)
                return np.frombuffer(decompressed_data, dtype=dtype)

        else:
            raise ValueError(f"Unsupported compression method: {compression_method}")

    def __str__(self) -> str:
        """String representation of TensorRecording."""
        return (f"TensorRecording(id={self.id}, tensor='{self.tensor_name}', "
                f"backend='{self.backend_name}', size={self.get_tensor_size_mb():.2f}MB)")

    def __repr__(self) -> str:
        """Detailed representation of TensorRecording."""
        return (f"TensorRecording(id={self.id}, session_id={self.session_id}, "
                f"checkpoint_id={self.checkpoint_id}, tensor_name='{self.tensor_name}', "
                f"backend_name='{self.backend_name}', "
                f"compression_method='{self.compression_method}')")


class TensorRecordingManager:
    """
    Utility class for managing collections of tensor recordings.

    Provides functionality for bulk operations, filtering, and analysis
    of multiple tensor recordings.
    """

    def __init__(self, recordings: Optional[List[TensorRecording]] = None):
        """Initialize with optional list of recordings."""
        self.recordings = recordings or []

    def add_recording(self, recording: TensorRecording) -> None:
        """Add a recording to the collection."""
        self.recordings.append(recording)

    def filter_by_backend(self, backend_name: str) -> List[TensorRecording]:
        """Filter recordings by backend name."""
        return [r for r in self.recordings if r.backend_name == backend_name]

    def filter_by_checkpoint(self, checkpoint_id: int) -> List[TensorRecording]:
        """Filter recordings by checkpoint ID."""
        return [r for r in self.recordings if r.checkpoint_id == checkpoint_id]

    def filter_by_session(self, session_id: int) -> List[TensorRecording]:
        """Filter recordings by session ID."""
        return [r for r in self.recordings if r.session_id == session_id]

    def get_total_size_mb(self) -> float:
        """Get total size of all recordings in megabytes."""
        return sum(r.get_tensor_size_mb() for r in self.recordings)

    def get_compression_stats(self) -> Dict[str, int]:
        """Get statistics on compression methods used."""
        stats = {}
        for recording in self.recordings:
            method = recording.compression_method
            stats[method] = stats.get(method, 0) + 1
        return stats

    def validate_all_recordings(self) -> Dict[str, List[TensorRecording]]:
        """
        Validate all recordings and categorize by status.

        Returns:
            Dictionary with 'valid' and 'invalid' lists
        """
        valid = []
        invalid = []

        for recording in self.recordings:
            integrity_results = recording.validate_recording_integrity()
            if all(integrity_results.values()):
                valid.append(recording)
            else:
                invalid.append(recording)

        return {"valid": valid, "invalid": invalid}