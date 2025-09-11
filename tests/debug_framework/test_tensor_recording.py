"""
Test module for TensorRecording model.

This test module validates the TensorRecording data model which provides
JSON-formatted capture of tensor data with complete metadata for offline validation scenarios.

TDD: This test MUST FAIL until the TensorRecording model is implemented.
"""

import pytest
import json
import os
import tempfile
import gzip
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.models.tensor_recording import TensorRecording
    TENSORRECORDING_AVAILABLE = True
except ImportError:
    TensorRecording = None
    TENSORRECORDING_AVAILABLE = False


class TestTensorRecording:
    """Test suite for TensorRecording model functionality."""

    def test_tensor_recording_import_fails(self):
        """Test that TensorRecording import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.models.tensor_recording import TensorRecording

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_tensor_recording_creation(self):
        """Test basic TensorRecording object creation."""
        tensor_metadata = {
            "dtype": "float32",
            "shape": [32, 128, 64],
            "strides": [8192, 64, 1],
            "device": "cuda:0",
            "memory_layout": "contiguous",
            "byte_order": "little"
        }
        device_info = {
            "platform": "cuda",
            "device_id": 0,
            "compute_capability": "8.0",
            "memory_total": "24GB"
        }
        
        recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="attention_output",
            tensor_metadata=tensor_metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="cuda_attention",
            device_info=device_info,
            compression_method="gzip"
        )
        
        assert recording.session_id == 1
        assert recording.checkpoint_id == 1
        assert recording.tensor_name == "attention_output"
        assert recording.tensor_metadata == tensor_metadata
        assert recording.tensor_data_path == "/path/to/tensor.bin"
        assert recording.backend_name == "cuda_attention"
        assert recording.device_info == device_info
        assert recording.compression_method == "gzip"
        assert recording.recording_timestamp is not None

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_tensor_metadata_validation(self):
        """Test tensor metadata must include required fields."""
        # Valid complete metadata
        valid_metadata = {
            "dtype": "float32",
            "shape": [32, 128, 64],
            "strides": [8192, 64, 1],
            "device": "cuda:0",
            "memory_layout": "contiguous",
            "byte_order": "little"
        }
        
        recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=valid_metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="test_backend",
            device_info={"platform": "cuda"}
        )
        assert recording.tensor_metadata == valid_metadata

        # Invalid metadata missing required fields
        invalid_metadata = {
            "dtype": "float32",
            "shape": [32, 128, 64],
            # Missing strides, device, memory_layout, byte_order
        }
        
        with pytest.raises(ValueError, match="tensor_metadata must include dtype, shape, strides, device information"):
            TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=invalid_metadata,
                tensor_data_path="/path/to/tensor.bin",
                backend_name="test_backend",
                device_info={"platform": "cuda"}
            )

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_tensor_data_path_validation(self):
        """Test tensor data path must point to valid binary file."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        # Test valid file path
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"tensor_data")
            temp_path = temp_file.name
        
        try:
            recording = TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path=temp_path,
                backend_name="test_backend",
                device_info=device_info
            )
            assert recording.tensor_data_path == temp_path
        finally:
            os.unlink(temp_path)

        # Test non-existent file path
        with pytest.raises(ValueError, match="tensor_data_path must point to valid binary file"):
            TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path="/nonexistent/file.bin",
                backend_name="test_backend",
                device_info=device_info
            )

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_backend_name_validation(self):
        """Test backend name must match registered backend."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        # Mock backend registry
        with patch('debug_framework.services.plugin_registry.PluginRegistry') as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry.return_value = mock_registry_instance
            mock_registry_instance.is_registered.return_value = True
            
            # Valid registered backend
            recording = TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path="/path/to/tensor.bin",
                backend_name="registered_backend",
                device_info=device_info
            )
            assert recording.backend_name == "registered_backend"

        # Mock unregistered backend
        with patch('debug_framework.services.plugin_registry.PluginRegistry') as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry.return_value = mock_registry_instance
            mock_registry_instance.is_registered.return_value = False
            
            with pytest.raises(ValueError, match="backend_name must match registered backend"):
                TensorRecording(
                    session_id=1,
                    checkpoint_id=1,
                    tensor_name="test_tensor",
                    tensor_metadata=metadata,
                    tensor_data_path="/path/to/tensor.bin",
                    backend_name="unregistered_backend",
                    device_info=device_info
                )

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_device_info_validation(self):
        """Test device info must include platform-specific details."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cuda:0", "memory_layout": "contiguous", "byte_order": "little"
        }
        
        # Valid CUDA device info
        cuda_device_info = {
            "platform": "cuda",
            "device_id": 0,
            "compute_capability": "8.0",
            "memory_total": "24GB",
            "driver_version": "11.7"
        }
        
        recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="cuda_backend",
            device_info=cuda_device_info
        )
        assert recording.device_info == cuda_device_info

        # Valid Metal device info
        metal_device_info = {
            "platform": "metal",
            "device_name": "Apple M2 Pro",
            "max_threads_per_group": 1024,
            "supports_family": "metal3"
        }
        
        recording2 = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="metal_backend",
            device_info=metal_device_info
        )
        assert recording2.device_info == metal_device_info

        # Invalid device info missing platform
        invalid_device_info = {
            "device_id": 0,
            "memory_total": "24GB"
        }
        
        with pytest.raises(ValueError, match="device_info must include platform-specific details"):
            TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path="/path/to/tensor.bin",
                backend_name="test_backend",
                device_info=invalid_device_info
            )

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_recording_timestamp_format(self):
        """Test recording timestamp must be in ISO format."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="test_backend",
            device_info=device_info
        )
        
        # Test that timestamp is automatically set and in ISO format
        assert recording.recording_timestamp is not None
        
        # Should be able to parse as ISO format (Python <3.11 compatibility)
        timestamp_str = recording.recording_timestamp
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'  # Convert Z to +00:00 for Python <3.11
        
        parsed_time = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed_time, datetime)

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_compression_method_validation(self):
        """Test compression method validation with conditional LZ4 support."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        # Test core compression methods (always available)
        core_methods = ["none", "gzip"]
        for method in core_methods:
            recording = TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path="/path/to/tensor.bin",
                backend_name="test_backend",
                device_info=device_info,
                compression_method=method
            )
            assert recording.compression_method == method

        # Test LZ4 compression (conditional on availability)
        try:
            import lz4
            LZ4_AVAILABLE = True
        except ImportError:
            LZ4_AVAILABLE = False
        
        if LZ4_AVAILABLE:
            recording_lz4 = TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path="/path/to/tensor.bin",
                backend_name="test_backend",
                device_info=device_info,
                compression_method="lz4"
            )
            assert recording_lz4.compression_method == "lz4"
        else:
            # Should raise error when LZ4 not available
            with pytest.raises(ValueError, match="LZ4 compression not available"):
                TensorRecording(
                    session_id=1,
                    checkpoint_id=1,
                    tensor_name="test_tensor",
                    tensor_metadata=metadata,
                    tensor_data_path="/path/to/tensor.bin",
                    backend_name="test_backend",
                    device_info=device_info,
                    compression_method="lz4"
                )

        # Test invalid compression method
        with pytest.raises(ValueError, match="Invalid compression method"):
            TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path="/path/to/tensor.bin",
                backend_name="test_backend",
                device_info=device_info,
                compression_method="invalid_compression"
            )

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_file_size_tracking(self):
        """Test file size tracking for tensor data files."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        # Create temporary file with known size
        test_data = b"tensor_data_12345"
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_data)
            temp_path = temp_file.name
            expected_size = len(test_data)
        
        try:
            recording = TensorRecording(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_metadata=metadata,
                tensor_data_path=temp_path,
                backend_name="test_backend",
                device_info=device_info
            )
            
            # File size should be automatically detected
            assert recording.file_size_bytes == expected_size
        finally:
            os.unlink(temp_path)

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_tensor_data_storage_and_retrieval(self):
        """Test storing and retrieving tensor data."""
        # Create test tensor
        test_tensor = np.random.randn(32, 64).astype(np.float32)
        metadata = {
            "dtype": "float32",
            "shape": list(test_tensor.shape),
            "strides": [64, 1],
            "device": "cpu",
            "memory_layout": "contiguous",
            "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test storing tensor data
            recording = TensorRecording.create_from_tensor(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor",
                tensor_data=test_tensor,
                backend_name="test_backend",
                device_info=device_info,
                storage_dir=temp_dir
            )
            
            assert recording.tensor_metadata["shape"] == list(test_tensor.shape)
            assert recording.tensor_metadata["dtype"] == "float32"
            assert os.path.exists(recording.tensor_data_path)
            
            # Test retrieving tensor data
            retrieved_tensor = recording.load_tensor_data()
            np.testing.assert_array_equal(retrieved_tensor, test_tensor)

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_compression_handling(self):
        """Test tensor data compression and decompression with deterministic data."""
        # Use deterministic, compressible data instead of random data
        test_tensor = np.zeros((100, 100), dtype=np.float32)  # Highly compressible
        # Add some pattern to make it realistic
        test_tensor[::10, ::10] = 1.0  # Sparse pattern
        
        device_info = {"platform": "cpu"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create uncompressed version for size comparison
            recording_none = TensorRecording.create_from_tensor(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor_uncompressed",
                tensor_data=test_tensor,
                backend_name="test_backend",
                device_info=device_info,
                storage_dir=temp_dir,
                compression_method="none"
            )
            
            # Test gzip compression
            recording_gzip = TensorRecording.create_from_tensor(
                session_id=1,
                checkpoint_id=1,
                tensor_name="test_tensor_compressed",
                tensor_data=test_tensor,
                backend_name="test_backend",
                device_info=device_info,
                storage_dir=temp_dir,
                compression_method="gzip"
            )
            
            assert recording_gzip.compression_method == "gzip"
            assert os.path.exists(recording_gzip.tensor_data_path)
            
            # Test compression effectiveness with deterministic data
            uncompressed_size = recording_none.file_size_bytes
            compressed_size = recording_gzip.file_size_bytes
            compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
            
            # With zeros + sparse pattern, should achieve good compression
            assert compression_ratio > 2.0, f"Expected compression ratio > 2.0, got {compression_ratio}"
            
            # Test data integrity after compression
            retrieved_tensor = recording_gzip.load_tensor_data()
            np.testing.assert_array_equal(retrieved_tensor, test_tensor)

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_database_relationships(self):
        """Test database relationships with DebugSession and ValidationCheckpoint."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {"platform": "cpu"}
        
        recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="test_backend",
            device_info=device_info
        )
        
        # Mock database operations
        with patch('debug_framework.services.database_manager.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            
            # Test save operation
            recording.save()
            mock_db_instance.insert_tensor_recording.assert_called_once()
            
            # Test foreign key relationships
            args = mock_db_instance.insert_tensor_recording.call_args[0][0]
            assert args['session_id'] == 1
            assert args['checkpoint_id'] == 1

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_offline_validation_workflow(self):
        """Test complete offline validation workflow."""
        device_info = {"platform": "cpu"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Record reference backend tensor
            reference_tensor = np.random.randn(32, 64).astype(np.float32)
            reference_recording = TensorRecording.create_from_tensor(
                session_id=1,
                checkpoint_id=1,
                tensor_name="attention_output",
                tensor_data=reference_tensor,
                backend_name="reference_backend",
                device_info=device_info,
                storage_dir=temp_dir
            )
            
            # Step 2: Record alternative backend tensor
            alternative_tensor = reference_tensor + np.random.normal(0, 1e-6, reference_tensor.shape).astype(np.float32)
            alternative_recording = TensorRecording.create_from_tensor(
                session_id=1,
                checkpoint_id=1,
                tensor_name="attention_output",
                tensor_data=alternative_tensor,
                backend_name="alternative_backend",
                device_info=device_info,
                storage_dir=temp_dir
            )
            
            # Step 3: Load recordings for comparison
            ref_data = reference_recording.load_tensor_data()
            alt_data = alternative_recording.load_tensor_data()
            
            # Verify data integrity for offline comparison
            np.testing.assert_array_equal(ref_data, reference_tensor)
            np.testing.assert_array_equal(alt_data, alternative_tensor)

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cuda:0", "memory_layout": "contiguous", "byte_order": "little"
        }
        device_info = {
            "platform": "cuda", "device_id": 0, "compute_capability": "8.0"
        }
        
        recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="cuda_backend",
            device_info=device_info,
            compression_method="gzip",
            file_size_bytes=1024
        )
        
        # Test serialization
        recording_dict = recording.to_dict()
        assert recording_dict["session_id"] == 1
        assert recording_dict["checkpoint_id"] == 1
        assert recording_dict["tensor_metadata"] == metadata
        assert recording_dict["device_info"] == device_info
        
        # Test deserialization
        restored_recording = TensorRecording.from_dict(recording_dict)
        assert restored_recording.session_id == recording.session_id
        assert restored_recording.tensor_metadata == recording.tensor_metadata
        assert restored_recording.device_info == recording.device_info

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_cross_platform_device_handling(self):
        """Test device info handling across different platforms."""
        metadata = {
            "dtype": "float32", "shape": [32, 64], "strides": [64, 1],
            "device": "cpu", "memory_layout": "contiguous", "byte_order": "little"
        }
        
        # CPU device info
        cpu_device_info = {
            "platform": "cpu",
            "processor": "Intel Xeon",
            "cores": 8,
            "memory_total": "32GB"
        }
        
        cpu_recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="cpu_backend",
            device_info=cpu_device_info
        )
        assert cpu_recording.get_platform() == "cpu"
        
        # Metal device info  
        metal_device_info = {
            "platform": "metal",
            "device_name": "Apple M2 Pro",
            "unified_memory": True,
            "max_buffer_length": "21GB"
        }
        
        metal_recording = TensorRecording(
            session_id=1,
            checkpoint_id=1,
            tensor_name="test_tensor",
            tensor_metadata=metadata,
            tensor_data_path="/path/to/tensor.bin",
            backend_name="metal_backend",
            device_info=metal_device_info
        )
        assert metal_recording.get_platform() == "metal"

    @pytest.mark.skipif(not TENSORRECORDING_AVAILABLE, reason="TensorRecording not implemented")
    def test_storage_optimization(self):
        """Test storage optimization features with deterministic data."""
        # Use structured data that compresses well instead of random data
        large_tensor = np.zeros((1000, 1000), dtype=np.float32)
        # Add structured pattern for realistic data that still compresses well
        large_tensor[::50, ::50] = 1.0  # Sparse grid pattern
        large_tensor[:100, :100] = 0.5  # Dense block
        
        device_info = {"platform": "cpu"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test storage with compression for large tensors
            recording = TensorRecording.create_from_tensor(
                session_id=1,
                checkpoint_id=1,
                tensor_name="large_tensor",
                tensor_data=large_tensor,
                backend_name="test_backend",
                device_info=device_info,
                storage_dir=temp_dir,
                compression_method="gzip"
            )
            
            # Verify compression mechanism works
            uncompressed_size = large_tensor.nbytes
            compressed_size = recording.file_size_bytes
            
            assert compressed_size > 0
            assert recording.compression_method == "gzip"
            
            # With structured pattern, compression should be effective
            compression_ratio = uncompressed_size / compressed_size
            assert compression_ratio > 5.0, f"Expected significant compression for structured data, got ratio {compression_ratio}"
            
            # Verify data integrity
            retrieved_tensor = recording.load_tensor_data()
            np.testing.assert_array_equal(retrieved_tensor, large_tensor)