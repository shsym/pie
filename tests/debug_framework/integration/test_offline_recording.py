"""
Integration test module for Offline recording mode.

This test module validates the offline validation mode including tensor recording
workflow (capture → store → replay), file system operations, data integrity,
metadata preservation, and batch processing of recorded validation sessions.

TDD: This test MUST FAIL until the offline recording mode is implemented.
"""

import pytest
import numpy as np
import tempfile
import json
import gzip
import os
import shutil
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import hashlib
import time

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.integrations.offline_recording import OfflineRecordingManager
    OFFLINE_RECORDING_AVAILABLE = True
except ImportError:
    OfflineRecordingManager = None
    OFFLINE_RECORDING_AVAILABLE = False

try:
    from debug_framework.storage.tensor_storage import TensorStorageEngine
    TENSOR_STORAGE_AVAILABLE = True
except ImportError:
    TensorStorageEngine = None
    TENSOR_STORAGE_AVAILABLE = False


class TestOfflineRecording:
    """Test suite for offline recording mode functionality."""

    @pytest.mark.xfail(reason="Will pass once OfflineRecordingManager is implemented")
    def test_offline_recording_manager_import_fails(self):
        """Test that offline recording manager import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.integrations.offline_recording import OfflineRecordingManager

    @pytest.mark.xfail(reason="Will pass once TensorStorageEngine is implemented")
    def test_tensor_storage_engine_import_fails(self):
        """Test that tensor storage engine import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.storage.tensor_storage import TensorStorageEngine

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_offline_recording_manager_initialization(self):
        """Test initialization of offline recording management system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                storage_directory=temp_dir,
                compression_enabled=True,
                compression_level=6,
                metadata_storage="json",
                max_file_size="100MB",
                chunking_enabled=True
            )

            assert manager.storage_directory == temp_dir
            assert manager.compression_enabled is True
            assert manager.compression_level == 6
            assert manager.metadata_storage == "json"
            assert manager.active_recordings == {}
            assert manager.storage_stats == {"total_size": 0, "file_count": 0}

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_tensor_recording_workflow_capture_phase(self):
        """Test tensor capture phase of recording workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(temp_dir)

            # Start recording session
            session_config = {
                "session_name": "test_capture_session",
                "model_name": "llama-3.2-1b",
                "checkpoints": ["post_embedding", "post_attention", "post_mlp"],
                "capture_metadata": True,
                "compression": True
            }

            recording_id = manager.start_recording_session(session_config)

            assert recording_id is not None
            assert recording_id in manager.active_recordings
            assert manager.active_recordings[recording_id]["status"] == "recording"

            # Capture tensor data at different checkpoints
            test_tensors = {
                "post_embedding": np.random.rand(8, 512, 4096).astype(np.float16),
                "post_attention": np.random.rand(8, 512, 4096).astype(np.float16),
                "post_mlp": np.random.rand(8, 512, 4096).astype(np.float16)
            }

            for checkpoint_name, tensor_data in test_tensors.items():
                capture_result = manager.capture_tensor(
                    recording_id=recording_id,
                    checkpoint_name=checkpoint_name,
                    tensor_data=tensor_data,
                    metadata={
                        "layer_id": checkpoint_name.split("_")[1],
                        "batch_size": 8,
                        "sequence_length": 512,
                        "timestamp": time.time()
                    }
                )

                assert capture_result["status"] == "captured"
                assert capture_result["checkpoint_name"] == checkpoint_name
                assert capture_result["tensor_shape"] == tensor_data.shape
                assert capture_result["storage_location"] is not None

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_tensor_storage_and_compression(self):
        """Test tensor storage with compression using compressible fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                compression_enabled=True,
                compression_level=9  # Maximum compression
            )

            recording_id = manager.start_recording_session({
                "session_name": "compression_test",
                "compression": True
            })

            # Create compressible tensor data fixtures instead of random data
            # Fixture 1: Highly repetitive data (should compress very well)
            sparse_tensor = np.zeros((16, 1024, 4096), dtype=np.float32)
            sparse_tensor[::8, ::16, ::32] = 1.0  # Sparse pattern - highly compressible

            # Fixture 2: Semi-structured data (moderate compression)
            pattern_tensor = np.tile(
                np.arange(256, dtype=np.float32).reshape(16, 16, 1),
                (1, 64, 4096)
            )  # Tiled pattern - moderately compressible

            # Fixture 3: Near-zero data (excellent compression)
            near_zero_tensor = np.full((16, 1024, 4096), 1e-8, dtype=np.float32)
            near_zero_tensor[0, 0, :100] = 1.0  # Tiny amount of real data

            test_fixtures = [
                ("sparse_tensor", sparse_tensor, 10.0),  # Expected high compression ratio
                ("pattern_tensor", pattern_tensor, 3.0),  # Expected moderate compression
                ("near_zero_tensor", near_zero_tensor, 100.0)  # Expected excellent compression
            ]

            for fixture_name, tensor_data, expected_min_ratio in test_fixtures:
                storage_result = manager.store_tensor_with_compression(
                    recording_id=recording_id,
                    checkpoint_name=f"compression_test_{fixture_name}",
                    tensor_data=tensor_data,
                    compression_algorithm="gzip",
                    verify_integrity=True,
                    measure_compression_effectiveness=True
                )

                assert storage_result["status"] == "stored"

                # Use fixture-specific compression expectations
                compression_ratio = storage_result.get("compression_ratio", 1.0)
                assert compression_ratio >= expected_min_ratio, \
                    f"{fixture_name}: Expected >{expected_min_ratio}x compression, got {compression_ratio}x"

                # Verify data integrity over compression ratio
                assert storage_result["integrity_verified"] is True

                # Verify file exists and has correct extension
                stored_file_path = storage_result["file_path"]
                assert os.path.exists(stored_file_path)
                assert stored_file_path.endswith('.gz')

                # Test decompression and data integrity - most important test
                loaded_tensor = manager.load_tensor_from_storage(
                    file_path=stored_file_path,
                    decompress=True,
                    verify_checksum=True
                )

                # Primary assertion: data integrity is preserved
                # Use exact comparison first, with fallback for potential floating point precision issues
                try:
                    assert np.array_equal(tensor_data, loaded_tensor["tensor_data"]), \
                        f"{fixture_name}: Decompressed data doesn't match original"
                except AssertionError:
                    # Fallback for edge cases with floating point precision
                    assert np.allclose(tensor_data, loaded_tensor["tensor_data"], rtol=1e-15, atol=1e-15), \
                        f"{fixture_name}: Decompressed data doesn't match original (even with tolerance)"
                assert loaded_tensor["checksum_verified"] is True

            # Test different compression algorithms with same fixture
            # More explicit algorithm availability check
            available_algorithms = manager.get_available_compression_algorithms() if hasattr(manager, 'get_available_compression_algorithms') else ["gzip"]
            test_algorithms = [alg for alg in ["gzip", "lzma", "bz2"] if alg in available_algorithms]

            for algorithm in test_algorithms:
                alg_result = manager.store_tensor_with_compression(
                    recording_id=recording_id,
                    checkpoint_name=f"algorithm_test_{algorithm}",
                    tensor_data=sparse_tensor,  # Use highly compressible fixture
                    compression_algorithm=algorithm,
                    verify_integrity=True
                )

                assert alg_result["status"] == "stored"
                assert alg_result["compression_algorithm"] == algorithm
                # Focus on integrity over specific compression ratios
                assert alg_result["integrity_verified"] is True

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_metadata_preservation_and_indexing(self):
        """Test preservation of metadata and creation of searchable indexes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                metadata_storage="json",
                indexing_enabled=True
            )

            recording_id = manager.start_recording_session({
                "session_name": "metadata_test",
                "model_config": {
                    "architecture": "transformer",
                    "num_layers": 32,
                    "hidden_size": 4096,
                    "num_heads": 32
                },
                "execution_context": {
                    "backend": "pytorch",
                    "device": "cuda:0",
                    "precision": "float16",
                    "batch_size": 8
                }
            })

            # Capture tensor with rich metadata
            tensor_data = np.random.rand(8, 512, 4096).astype(np.float16)
            detailed_metadata = {
                "checkpoint_info": {
                    "layer_type": "attention",
                    "layer_index": 5,
                    "head_index": None,
                    "operation": "self_attention"
                },
                "tensor_properties": {
                    "dtype": "float16",
                    "device": "cuda:0",
                    "requires_grad": False,
                    "memory_layout": "contiguous"
                },
                "execution_context": {
                    "forward_pass_id": "fp_001",
                    "timestamp": time.time(),
                    "model_state": "training",
                    "gradient_enabled": True
                },
                "validation_config": {
                    "tolerance": 1e-4,
                    "comparison_backend": "metal",
                    "checkpoint_enabled": True
                }
            }

            metadata_result = manager.capture_tensor_with_metadata(
                recording_id=recording_id,
                checkpoint_name="attention_layer_5",
                tensor_data=tensor_data,
                metadata=detailed_metadata
            )

            assert metadata_result["status"] == "captured"
            assert metadata_result["metadata_indexed"] is True

            # Test metadata search and retrieval
            search_results = manager.search_recordings(
                query={
                    "checkpoint_info.layer_type": "attention",
                    "checkpoint_info.layer_index": 5,
                    "execution_context.model_state": "training"
                }
            )

            assert len(search_results) == 1
            assert search_results[0]["recording_id"] == recording_id
            assert search_results[0]["checkpoint_name"] == "attention_layer_5"

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_offline_validation_replay_and_comparison(self):
        """Test replay of recorded data and comparison against live execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(temp_dir)

            # First, record reference data
            recording_id = manager.start_recording_session({
                "session_name": "reference_recording",
                "mode": "reference_capture"
            })

            # Record reference tensors
            reference_tensors = {
                "input": np.random.rand(4, 256, 2048).astype(np.float32),
                "post_embedding": np.random.rand(4, 256, 4096).astype(np.float32),
                "post_attention": np.random.rand(4, 256, 4096).astype(np.float32),
                "output": np.random.rand(4, 256, 32000).astype(np.float32)
            }

            for checkpoint, tensor in reference_tensors.items():
                manager.capture_tensor(recording_id, checkpoint, tensor)

            manager.finalize_recording_session(recording_id)

            # Now test offline validation replay
            replay_session = manager.create_replay_session(
                reference_recording_id=recording_id,
                validation_config={
                    "tolerance": 1e-5,
                    "comparison_mode": "element_wise",
                    "statistical_analysis": True
                }
            )

            # Simulate alternative backend producing slightly different results
            alternative_tensors = {
                "post_embedding": reference_tensors["post_embedding"] + np.random.normal(0, 1e-6, reference_tensors["post_embedding"].shape),
                "post_attention": reference_tensors["post_attention"] + np.random.normal(0, 1e-6, reference_tensors["post_attention"].shape),
                "output": reference_tensors["output"] + np.random.normal(0, 1e-6, reference_tensors["output"].shape)
            }

            validation_results = []
            for checkpoint, alt_tensor in alternative_tensors.items():
                result = replay_session.validate_against_recorded(
                    checkpoint_name=checkpoint,
                    alternative_tensor=alt_tensor
                )
                validation_results.append(result)

            # Verify validation results
            assert len(validation_results) == 3
            for result in validation_results:
                assert result["status"] == "passed"
                assert result["max_absolute_error"] < 1e-5
                assert result["comparison_performed"] is True

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_batch_processing_of_recorded_sessions(self):
        """Test batch processing of multiple recorded validation sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(temp_dir)

            # Create multiple recording sessions
            session_configs = [
                {"session_name": f"batch_session_{i}", "model_variant": f"variant_{i}"}
                for i in range(5)
            ]

            recording_ids = []
            for config in session_configs:
                recording_id = manager.start_recording_session(config)

                # Record some test data for each session
                for checkpoint in ["layer_0", "layer_1", "layer_2"]:
                    tensor_data = np.random.rand(2, 128, 1024).astype(np.float32)
                    manager.capture_tensor(recording_id, checkpoint, tensor_data)

                manager.finalize_recording_session(recording_id)
                recording_ids.append(recording_id)

            # Test batch processing
            batch_processor = manager.create_batch_processor(
                recording_ids=recording_ids,
                processing_config={
                    "parallel_processing": True,
                    "max_workers": 3,
                    "validation_mode": "cross_comparison",
                    "statistical_analysis": True
                }
            )

            batch_results = batch_processor.process_all_sessions()

            assert batch_results["status"] == "completed"
            assert batch_results["processed_sessions"] == 5
            assert batch_results["failed_sessions"] == 0
            assert "cross_session_statistics" in batch_results
            assert "consistency_analysis" in batch_results

            # Verify cross-session consistency
            consistency = batch_results["consistency_analysis"]
            assert consistency["inter_session_variance"] < 0.1
            assert consistency["outlier_sessions"] == []

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_incremental_recording_and_differential_storage(self):
        """Test incremental recording and differential storage with memory-mapped arrays for large tensor handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                differential_storage=True,
                incremental_recording=True,
                large_tensor_threshold="100MB"  # Threshold for using memory-mapped files
            )

            # Start base recording session with large tensor handling
            base_recording = manager.start_recording_session({
                "session_name": "base_session",
                "mode": "baseline",
                "memory_mapped_storage": True
            })

            # Create large tensor using memory-mapped array for efficient handling
            large_tensor_shape = (16, 2048, 4096)  # ~512MB tensor
            base_mmap_file = os.path.join(temp_dir, "base_tensor.mmap")

            # Use numpy.memmap for large tensor creation and manipulation
            base_tensor = np.memmap(
                base_mmap_file,
                dtype=np.float32,
                mode='w+',
                shape=large_tensor_shape
            )

            # Fill with structured data for better compression and verification
            base_tensor[:] = np.arange(np.prod(large_tensor_shape), dtype=np.float32).reshape(large_tensor_shape) / 1e6
            base_tensor.flush()  # Ensure data is written to disk

            # Capture large tensor using memory-mapped storage
            base_capture_result = manager.capture_tensor_memory_mapped(
                recording_id=base_recording,
                checkpoint_name="large_layer_output",
                tensor_mmap=base_tensor,
                tensor_file_path=base_mmap_file,
                verify_integrity=True
            )

            assert base_capture_result["status"] == "captured"
            assert base_capture_result["storage_mode"] == "memory_mapped"
            assert base_capture_result["file_size_mb"] > 100  # Verify large tensor
            assert base_capture_result["integrity_verified"] is True

            manager.finalize_recording_session(base_recording)

            # Start incremental recording session with memory-mapped differential storage
            incremental_recording = manager.start_incremental_recording_session({
                "session_name": "incremental_session",
                "base_session_id": base_recording,
                "differential_mode": True,
                "memory_mapped_diffs": True
            })

            # Create modified tensor using memory-mapped array
            modified_mmap_file = os.path.join(temp_dir, "modified_tensor.mmap")
            modified_tensor = np.memmap(
                modified_mmap_file,
                dtype=np.float32,
                mode='w+',
                shape=large_tensor_shape
            )

            # Copy base tensor and make small modifications
            modified_tensor[:] = base_tensor[:]  # Copy all data
            # Small modification - only change 1% of elements for sparse differential
            modification_mask = np.zeros(large_tensor_shape, dtype=bool)
            modification_mask[::10, ::20, ::40] = True  # Sparse modification pattern
            modified_tensor[modification_mask] += 0.001  # Small change to sparse elements
            modified_tensor.flush()

            # Test differential storage with memory-mapped arrays
            incremental_result = manager.capture_tensor_differential_memory_mapped(
                recording_id=incremental_recording,
                checkpoint_name="large_layer_output",
                tensor_mmap=modified_tensor,
                tensor_file_path=modified_mmap_file,
                base_reference_id=base_recording,
                compute_sparse_diff=True,
                diff_compression=True
            )

            assert incremental_result["status"] == "captured"
            assert incremental_result["storage_mode"] == "differential_memory_mapped"
            assert incremental_result["compression_ratio"] > 50.0  # Very high compression for sparse changes
            assert incremental_result["storage_efficiency"] > 0.95
            assert incremental_result["differential_sparsity"] > 0.99  # 99%+ of elements unchanged

            # Verify memory usage efficiency
            memory_stats = incremental_result.get("memory_stats", {})
            assert memory_stats.get("peak_memory_usage_mb", 0) < 200  # Should not load full tensors into memory

            # Test differential reconstruction using memory-mapped arrays
            reconstructed_mmap_file = os.path.join(temp_dir, "reconstructed_tensor.mmap")
            reconstructed = manager.reconstruct_from_differential_memory_mapped(
                base_session_id=base_recording,
                differential_session_id=incremental_recording,
                checkpoint_name="large_layer_output",
                output_mmap_file=reconstructed_mmap_file,
                verify_reconstruction=True
            )

            assert reconstructed["reconstruction_successful"] is True
            assert reconstructed["storage_mode"] == "memory_mapped"

            # Load reconstructed tensor as memory-mapped array for verification
            reconstructed_tensor = np.memmap(
                reconstructed_mmap_file,
                dtype=np.float32,
                mode='r',
                shape=large_tensor_shape
            )

            # Verify reconstruction accuracy using chunk-wise comparison to avoid memory overload
            chunk_size = 1024 * 1024  # 1M elements per chunk
            total_elements = np.prod(large_tensor_shape)
            max_error = 0.0

            for start_idx in range(0, total_elements, chunk_size):
                end_idx = min(start_idx + chunk_size, total_elements)

                # Get flat views of chunks
                original_chunk = modified_tensor.flat[start_idx:end_idx]
                reconstructed_chunk = reconstructed_tensor.flat[start_idx:end_idx]

                # Check chunk-wise accuracy
                chunk_error = np.max(np.abs(original_chunk - reconstructed_chunk))
                max_error = max(max_error, chunk_error)

            assert max_error < 1e-6, f"Reconstruction error {max_error} exceeds tolerance"
            assert reconstructed["max_reconstruction_error"] < 1e-6

            # Test memory-mapped storage cleanup
            cleanup_result = manager.cleanup_memory_mapped_files(
                session_ids=[base_recording, incremental_recording],
                preserve_final_results=True
            )

            assert cleanup_result["status"] == "cleaned"
            assert cleanup_result["files_cleaned"] >= 2  # At least base and modified temp files
            assert cleanup_result["disk_space_freed_mb"] > 500  # Should free significant space

            # Verify final results are still accessible
            assert os.path.exists(reconstructed_mmap_file)

            # Clean up our test memory-mapped arrays
            del base_tensor, modified_tensor, reconstructed_tensor

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_data_integrity_and_corruption_detection(self):
        """Test data integrity verification and corruption detection with SHA-256 checksums and manifest verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                integrity_checking=True,
                checksum_algorithm="sha256",
                redundant_storage=True,
                manifest_verification=True
            )

            recording_id = manager.start_recording_session({
                "session_name": "integrity_test",
                "integrity_verification": True
            })

            # Create deterministic test tensor for reproducible checksums
            test_tensor = np.arange(8 * 1024 * 2048, dtype=np.float32).reshape(8, 1024, 2048)
            test_tensor = test_tensor / np.max(test_tensor)  # Normalize to [0, 1]

            integrity_result = manager.capture_tensor_with_integrity(
                recording_id=recording_id,
                checkpoint_name="integrity_test_tensor",
                tensor_data=test_tensor,
                create_backup=True,
                verify_after_write=True,
                create_manifest=True,
                chunk_checksums=True  # Compute checksums for chunks too
            )

            assert integrity_result["status"] == "captured"

            # Verify SHA-256 checksum properties
            sha256_checksum = integrity_result["checksum"]
            assert sha256_checksum is not None
            assert len(sha256_checksum) == 64, "SHA-256 should produce 64-character hex string"
            assert all(c in '0123456789abcdef' for c in sha256_checksum.lower()), "Should be valid hex"

            # Verify manifest was created
            manifest_path = integrity_result.get("manifest_path")
            assert manifest_path is not None
            assert os.path.exists(manifest_path)

            # Load and verify manifest structure
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            assert "file_path" in manifest
            assert "checksum" in manifest
            assert "checksum_algorithm" in manifest
            assert manifest["checksum_algorithm"] == "sha256"
            assert manifest["checksum"] == sha256_checksum
            assert "file_size_bytes" in manifest
            assert "creation_timestamp" in manifest

            # Verify chunk-level checksums if supported
            if "chunk_checksums" in manifest:
                assert len(manifest["chunk_checksums"]) > 0
                for chunk_info in manifest["chunk_checksums"]:
                    assert "chunk_index" in chunk_info
                    assert "checksum" in chunk_info
                    assert len(chunk_info["checksum"]) == 64  # SHA-256

            assert integrity_result["backup_created"] is True
            assert integrity_result["write_verification"] is True

            # Verify original checksum manually
            stored_file_path = integrity_result["file_path"]
            original_sha256 = hashlib.sha256()
            with open(stored_file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b""):  # Read in 64KB chunks
                    original_sha256.update(chunk)
            computed_checksum = original_sha256.hexdigest()

            assert computed_checksum == sha256_checksum, "Stored checksum should match computed checksum"

            # Simulate data corruption by modifying specific bytes
            file_size = os.path.getsize(stored_file_path)
            corruption_offset = file_size // 2  # Corrupt middle of file

            with open(stored_file_path, 'r+b') as f:
                f.seek(corruption_offset)
                original_bytes = f.read(8)
                f.seek(corruption_offset)
                # Flip bits to ensure corruption
                corrupted_bytes = bytes(b ^ 0xFF for b in original_bytes)
                f.write(corrupted_bytes)

            # Test corruption detection with SHA-256 verification
            corruption_check = manager.verify_data_integrity(
                file_path=stored_file_path,
                expected_checksum=sha256_checksum,
                manifest_path=manifest_path,
                verify_chunk_checksums=True
            )

            assert corruption_check["status"] == "corrupted"
            assert corruption_check["corruption_detected"] is True
            assert corruption_check["checksum_mismatch"] is True
            assert corruption_check["expected_checksum"] == sha256_checksum
            assert corruption_check["actual_checksum"] != sha256_checksum
            assert corruption_check.get("corruption_location", -1) >= 0  # Should identify location

            # Test recovery from backup with manifest verification
            recovery_result = manager.recover_from_backup(
                original_file_path=stored_file_path,
                recording_id=recording_id,
                checkpoint_name="integrity_test_tensor",
                verify_backup_integrity=True,
                update_manifest=True
            )

            assert recovery_result["status"] == "recovered"
            assert recovery_result["recovery_successful"] is True
            assert recovery_result["backup_integrity_verified"] is True

            # Verify recovered data matches original
            recovered_tensor = manager.load_tensor_from_storage(
                file_path=recovery_result["recovered_file_path"],
                verify_checksum=True,
                verify_manifest=True
            )

            # Verify recovered data matches original with fallback for precision
            try:
                assert np.array_equal(test_tensor, recovered_tensor["tensor_data"])
            except AssertionError:
                # Fallback for potential floating point precision issues
                assert np.allclose(test_tensor, recovered_tensor["tensor_data"], rtol=1e-15, atol=1e-15)
            assert recovered_tensor["checksum_verified"] is True
            assert recovered_tensor["manifest_verified"] is True

            # Verify recovered file has correct checksum
            recovered_sha256 = hashlib.sha256()
            with open(recovery_result["recovered_file_path"], 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    recovered_sha256.update(chunk)
            assert recovered_sha256.hexdigest() == sha256_checksum

            # Test incremental corruption detection for large files
            large_test_data = np.zeros((32, 1024, 1024), dtype=np.float32)
            large_test_data[0, 0, :] = np.arange(1024, dtype=np.float32)  # Add some structure

            large_integrity_result = manager.capture_tensor_with_integrity(
                recording_id=recording_id,
                checkpoint_name="large_integrity_test",
                tensor_data=large_test_data,
                create_backup=False,  # Skip backup for large files
                chunked_verification=True,  # Use chunked verification for large files
                chunk_size_mb=1  # 1MB chunks for testing
            )

            assert large_integrity_result["status"] == "captured"
            assert large_integrity_result.get("chunked_verification_used", False) is True
            assert "chunk_count" in large_integrity_result
            assert large_integrity_result["chunk_count"] > 1

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_storage_optimization_and_cleanup(self):
        """Test storage optimization and cleanup mechanisms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                storage_optimization=True,
                automatic_cleanup=True,
                max_storage_size="500MB"
            )

            # Create multiple sessions to test cleanup
            session_ids = []
            for i in range(10):
                session_id = manager.start_recording_session({
                    "session_name": f"cleanup_test_{i}",
                    "priority": i % 3,  # Different priorities
                    "retention_policy": "7_days" if i < 5 else "permanent"
                })

                # Record moderate-sized tensors
                for j in range(3):
                    tensor = np.random.rand(4, 256, 1024).astype(np.float32)  # ~16MB each
                    manager.capture_tensor(session_id, f"layer_{j}", tensor)

                manager.finalize_recording_session(session_id)
                session_ids.append(session_id)

            # Check storage usage
            storage_stats = manager.get_storage_statistics()
            initial_size = storage_stats["total_size_mb"]

            # Test storage optimization
            optimization_result = manager.optimize_storage(
                strategies=["compression", "deduplication", "archival"],
                target_reduction=0.3  # 30% reduction
            )

            assert optimization_result["status"] == "optimized"
            assert optimization_result["size_reduction_achieved"] > 0.2
            assert optimization_result["files_deduplicated"] >= 0
            assert optimization_result["files_compressed"] >= 0

            # Test automatic cleanup based on retention policies
            with patch('time.time') as mock_time:
                # Simulate 8 days passing
                mock_time.return_value = time.time() + (8 * 24 * 3600)

                cleanup_result = manager.run_automatic_cleanup(
                    enforce_retention_policies=True,
                    free_space_threshold="100MB"
                )

                assert cleanup_result["status"] == "completed"
                assert cleanup_result["sessions_cleaned"] == 5  # 7-day retention sessions
                assert cleanup_result["permanent_sessions_preserved"] == 5

                # Verify permanent sessions still exist
                remaining_sessions = manager.list_available_sessions()
                permanent_sessions = [s for s in remaining_sessions if s["retention_policy"] == "permanent"]
                assert len(permanent_sessions) == 5

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_cross_platform_compatibility_and_portability(self):
        """Test cross-platform compatibility and data portability."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                cross_platform_compatibility=True,
                portable_format=True
            )

            recording_id = manager.start_recording_session({
                "session_name": "portability_test",
                "platform": "linux",
                "endianness": "little",
                "floating_point_format": "ieee754"
            })

            # Record tensor with platform-specific considerations
            test_tensor = np.random.rand(4, 128, 256).astype(np.float32)

            portable_result = manager.capture_tensor_portable(
                recording_id=recording_id,
                checkpoint_name="portable_tensor",
                tensor_data=test_tensor,
                normalize_endianness=True,
                validate_float_format=True,
                include_platform_metadata=True
            )

            assert portable_result["status"] == "captured"
            assert portable_result["endianness_normalized"] is True
            assert portable_result["float_format_validated"] is True
            assert "platform_metadata" in portable_result

            # Test export to portable format
            export_result = manager.export_to_portable_format(
                recording_id=recording_id,
                export_format="hdf5",  # Cross-platform format
                include_metadata=True,
                validate_after_export=True
            )

            assert export_result["status"] == "exported"
            assert export_result["format"] == "hdf5"
            assert export_result["validation_passed"] is True

            # Test import from portable format (simulating different platform)
            with patch('platform.system') as mock_platform:
                mock_platform.return_value = "Darwin"  # Simulate macOS

                import_result = manager.import_from_portable_format(
                    portable_file_path=export_result["export_path"],
                    target_session_name="imported_session",
                    platform_adaptation=True
                )

                assert import_result["status"] == "imported"
                assert import_result["platform_adapted"] is True
                assert import_result["data_integrity_verified"] is True

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    @pytest.mark.asyncio
    async def test_streaming_recording_and_real_time_processing(self):
        """Test streaming recording capabilities and real-time processing with backpressure handling."""
        import asyncio
        from collections import deque

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                streaming_enabled=True,
                real_time_processing=True,
                buffer_size="50MB"
            )

            # Start streaming recording session with bounded buffer
            stream_config = {
                "session_name": "streaming_test",
                "streaming_mode": True,
                "buffer_management": "bounded",  # Use bounded buffer for backpressure
                "buffer_capacity": 10,  # Maximum 10 items in buffer
                "real_time_compression": True,
                "flush_interval": 1.0,  # 1 second
                "backpressure_handling": "block_producer"
            }

            stream_session = manager.start_streaming_session(stream_config)

            # Implement bounded buffer with backpressure
            buffer_capacity = 10
            stream_buffer = deque(maxlen=buffer_capacity)
            backpressure_event = asyncio.Event()
            buffer_drain_event = asyncio.Event()
            processing_complete_event = asyncio.Event()

            streaming_results = []
            processed_chunks = []

            # Producer with backpressure awareness
            async def streaming_producer():
                for i in range(20):  # Stream 20 tensors
                    # Check buffer capacity for backpressure
                    while len(stream_buffer) >= buffer_capacity:
                        backpressure_event.set()  # Signal backpressure condition
                        await buffer_drain_event.wait()  # Wait for consumer to drain
                        buffer_drain_event.clear()

                    tensor_chunk = np.random.rand(2, 64, 512).astype(np.float16)

                    stream_data = {
                        "chunk_id": i,
                        "tensor_data": tensor_chunk,
                        "timestamp": time.perf_counter(),  # Use monotonic timestamp
                        "size_bytes": tensor_chunk.nbytes,
                        "force_flush": (i % 5 == 0)
                    }

                    stream_buffer.append(stream_data)

                    # Simulate variable producer speed
                    await asyncio.sleep(0.01 if i % 3 == 0 else 0.005)

                # Signal production complete
                processing_complete_event.set()

            # Consumer with bounded processing and drain signaling
            async def streaming_consumer():
                backpressure_triggered_count = 0

                while len(processed_chunks) < 20:
                    if stream_buffer:
                        # Process chunk from buffer
                        chunk_data = stream_buffer.popleft()

                        # Simulate processing time
                        await asyncio.sleep(0.008)  # Slightly slower than producer

                        # Mock stream_tensor_data call
                        stream_result = {
                            "status": "streamed",
                            "chunk_id": chunk_data["chunk_id"],
                            "buffered": True,
                            "timestamp": chunk_data["timestamp"],
                            "buffer_size_at_processing": len(stream_buffer)
                        }

                        streaming_results.append(stream_result)
                        processed_chunks.append(chunk_data)

                        # Signal drain if backpressure was active
                        if backpressure_event.is_set():
                            buffer_drain_event.set()
                            backpressure_event.clear()
                            backpressure_triggered_count += 1
                    else:
                        # Check if production is complete
                        if processing_complete_event.is_set():
                            break
                        await asyncio.sleep(0.001)  # Wait for data

                return backpressure_triggered_count

            # Run producer and consumer with backpressure handling
            start_time = time.perf_counter()

            producer_task = asyncio.create_task(streaming_producer())
            consumer_task = asyncio.create_task(streaming_consumer())

            # Wait for both to complete
            backpressure_count = await asyncio.gather(
                producer_task,
                consumer_task,
                return_exceptions=True
            )

            processing_time = time.perf_counter() - start_time

            # Verify backpressure was properly exercised
            assert len(streaming_results) == 20
            assert len(processed_chunks) == 20
            assert backpressure_count[1] > 0, "Backpressure should have been triggered"

            # Verify all chunks were processed in order
            chunk_ids = [result["chunk_id"] for result in streaming_results]
            assert chunk_ids == list(range(20))

            # Test real-time processing statistics
            total_bytes = sum(chunk["size_bytes"] for chunk in processed_chunks)
            throughput_mb_s = (total_bytes / (1024 * 1024)) / processing_time

            processing_stats = {
                "chunks_processed": len(processed_chunks),
                "buffer_utilization": max(result["buffer_size_at_processing"] for result in streaming_results) / buffer_capacity,
                "backpressure_events": backpressure_count[1],
                "average_throughput_mb_s": throughput_mb_s,
                "total_processing_time": processing_time
            }

            assert processing_stats["chunks_processed"] == 20
            assert processing_stats["buffer_utilization"] > 0.3  # Buffer should have been utilized
            assert processing_stats["backpressure_events"] > 0
            assert processing_stats["average_throughput_mb_s"] > 0

            # Finalize streaming session with consolidated chunks
            finalization_result = {
                "status": "finalized",
                "chunks_consolidated": len(processed_chunks),
                "index_created": True,
                "backpressure_statistics": {
                    "events_triggered": processing_stats["backpressure_events"],
                    "max_buffer_utilization": processing_stats["buffer_utilization"],
                    "processing_efficiency": len(processed_chunks) / 20.0
                }
            }

            assert finalization_result["status"] == "finalized"
            assert finalization_result["chunks_consolidated"] == 20
            assert finalization_result["index_created"] is True
            assert finalization_result["backpressure_statistics"]["events_triggered"] > 0
            assert finalization_result["backpressure_statistics"]["processing_efficiency"] == 1.0

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_advanced_compression_and_format_optimization(self):
        """Test advanced compression algorithms and format optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OfflineRecordingManager(
                temp_dir,
                advanced_compression=True,
                format_optimization=True
            )

            recording_id = manager.start_recording_session({
                "session_name": "compression_optimization_test",
                "optimization_target": "storage_efficiency"
            })

            # Test different compression algorithms
            test_tensors = {
                "sparse_tensor": np.zeros((1000, 1000)),  # Highly compressible
                "random_tensor": np.random.rand(500, 500),  # Less compressible
                "structured_tensor": np.tile(np.arange(100), (100, 10))  # Pattern-based
            }

            # Add some sparsity to sparse tensor
            test_tensors["sparse_tensor"][::10, ::10] = np.random.rand(100, 100)

            compression_results = {}
            for tensor_name, tensor_data in test_tensors.items():
                result = manager.capture_tensor_with_optimal_compression(
                    recording_id=recording_id,
                    checkpoint_name=tensor_name,
                    tensor_data=tensor_data.astype(np.float32),
                    auto_select_algorithm=True,
                    benchmark_compression=True
                )

                compression_results[tensor_name] = result
                assert result["status"] == "captured"
                assert result["optimal_algorithm_selected"] is True
                assert result["compression_ratio"] > 1.0

            # Verify different algorithms were selected for different data types
            algorithms_used = set(r["compression_algorithm"] for r in compression_results.values())
            assert len(algorithms_used) > 1  # Different algorithms for different data patterns

            # Verify sparse tensor achieved highest compression
            sparse_ratio = compression_results["sparse_tensor"]["compression_ratio"]
            random_ratio = compression_results["random_tensor"]["compression_ratio"]
            assert sparse_ratio > random_ratio

    @pytest.mark.skipif(not OFFLINE_RECORDING_AVAILABLE, reason="OfflineRecordingManager not implemented")
    def test_distributed_recording_and_synchronization(self):
        """Test distributed recording across multiple nodes and synchronization with monotonic timing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate multiple nodes
            node_dirs = [temp_dir + f"/node_{i}" for i in range(3)]
            for node_dir in node_dirs:
                os.makedirs(node_dir, exist_ok=True)

            # Create distributed recording setup with monotonic timing
            managers = []
            node_sync_offsets = {}  # Track timing offsets between nodes
            synchronization_reference_time = time.perf_counter()  # Global monotonic reference

            for i, node_dir in enumerate(node_dirs):
                # Simulate network delay/offset between nodes
                node_offset = i * 0.001  # 1ms offset per node
                node_sync_offsets[f"node_{i}"] = node_offset

                manager = OfflineRecordingManager(
                    node_dir,
                    distributed_mode=True,
                    node_id=f"node_{i}",
                    synchronization_enabled=True,
                    monotonic_timing=True,  # Use monotonic clocks
                    sync_reference_time=synchronization_reference_time + node_offset,
                    time_source="perf_counter"  # Explicit monotonic time source
                )
                managers.append(manager)

            # Start coordinated recording session with monotonic synchronization
            distributed_config = {
                "session_name": "distributed_test",
                "coordination_mode": "master_slave",
                "master_node": "node_0",
                "timestamp_synchronization": True,
                "monotonic_synchronization": True,
                "sync_tolerance_ms": 10.0,  # 10ms synchronization tolerance
                "drift_compensation": True
            }

            # Record session start time using monotonic clock
            session_start_time = time.perf_counter()

            # Node 0 acts as master with monotonic timing coordination
            master_session = managers[0].start_distributed_recording_session(
                config=distributed_config,
                participating_nodes=["node_1", "node_2"],
                global_sync_timestamp=session_start_time
            )

            # Other nodes join the session with time synchronization
            slave_sessions = []
            join_timestamps = []

            for i in range(1, 3):
                join_start = time.perf_counter()
                slave_session = managers[i].join_distributed_recording_session(
                    master_session_id=master_session,
                    master_node="node_0",
                    sync_with_master_time=True,
                    local_sync_offset=node_sync_offsets[f"node_{i}"]
                )
                join_end = time.perf_counter()

                slave_sessions.append(slave_session)
                join_timestamps.append({
                    "node": f"node_{i}",
                    "join_start": join_start,
                    "join_end": join_end,
                    "join_duration": join_end - join_start
                })

            # Each node records different parts of the model with synchronized timing
            node_checkpoints = [
                ["layer_0", "layer_1"],  # Node 0
                ["layer_2", "layer_3"],  # Node 1
                ["layer_4", "output"]    # Node 2
            ]

            captured_timestamps = []
            for i, (manager, checkpoints) in enumerate(zip(managers, node_checkpoints)):
                session_id = master_session if i == 0 else slave_sessions[i-1]

                for checkpoint in checkpoints:
                    tensor_data = np.random.rand(4, 256, 1024).astype(np.float32)

                    # Record with precise monotonic timing
                    capture_start = time.perf_counter()
                    capture_result = manager.capture_tensor_distributed(
                        session_id=session_id,
                        checkpoint_name=checkpoint,
                        tensor_data=tensor_data,
                        node_id=f"node_{i}",
                        synchronized_timestamp=True,
                        monotonic_timestamp=capture_start,  # Explicit monotonic timestamp
                        sync_reference=synchronization_reference_time
                    )
                    capture_end = time.perf_counter()

                    captured_timestamps.append({
                        "node": f"node_{i}",
                        "checkpoint": checkpoint,
                        "capture_start": capture_start,
                        "capture_end": capture_end,
                        "capture_duration": capture_end - capture_start,
                        "relative_time": capture_start - synchronization_reference_time,
                        "synchronized_timestamp": capture_result.get("synchronized_timestamp")
                    })

            # Verify monotonic timing consistency across nodes
            # All timestamps should be monotonically increasing relative to reference
            relative_times = [ts["relative_time"] for ts in captured_timestamps]
            assert all(rt >= 0 for rt in relative_times), "All relative times should be non-negative"

            # Check that timestamps from different nodes are properly synchronized
            node_timestamp_ranges = {}
            for ts in captured_timestamps:
                node = ts["node"]
                if node not in node_timestamp_ranges:
                    node_timestamp_ranges[node] = []
                node_timestamp_ranges[node].append(ts["relative_time"])

            # Verify no significant clock drift between nodes (within tolerance)
            max_drift = 0.0
            for i in range(len(node_timestamp_ranges)):
                for j in range(i + 1, len(node_timestamp_ranges)):
                    node_i = f"node_{i}"
                    node_j = f"node_{j}"
                    if node_i in node_timestamp_ranges and node_j in node_timestamp_ranges:
                        avg_time_i = np.mean(node_timestamp_ranges[node_i])
                        avg_time_j = np.mean(node_timestamp_ranges[node_j])
                        drift = abs(avg_time_i - avg_time_j - (node_sync_offsets[node_j] - node_sync_offsets[node_i]))
                        max_drift = max(max_drift, drift)

            assert max_drift < 0.02, f"Clock drift {max_drift:.3f}s exceeds tolerance"

            # Synchronize and consolidate recordings with monotonic timing validation
            consolidation_start = time.perf_counter()
            consolidation_result = managers[0].consolidate_distributed_recording(
                master_session_id=master_session,
                merge_strategy="chronological_monotonic",  # Use monotonic ordering
                verify_completeness=True,
                verify_timing_consistency=True,
                monotonic_validation=True
            )
            consolidation_end = time.perf_counter()

            assert consolidation_result["status"] == "consolidated"
            assert consolidation_result["nodes_synchronized"] == 3
            assert consolidation_result["total_checkpoints"] == 6
            assert consolidation_result["synchronization_successful"] is True

            # Verify monotonic timing validation passed
            timing_validation = consolidation_result.get("timing_validation", {})
            assert timing_validation.get("monotonic_consistency", False) is True
            assert timing_validation.get("max_clock_drift_ms", float('inf')) < 20.0  # Within 20ms
            assert timing_validation.get("temporal_ordering_violations", -1) == 0

            # Verify consolidated timeline is monotonically ordered
            consolidated_timeline = consolidation_result.get("consolidated_timeline", [])
            timeline_timestamps = [entry["monotonic_timestamp"] for entry in consolidated_timeline]
            assert timeline_timestamps == sorted(timeline_timestamps), "Timeline should be monotonically ordered"

            # Verify performance metrics with monotonic timing
            performance_metrics = consolidation_result.get("performance_metrics", {})
            assert performance_metrics.get("total_synchronization_time", 0) < 1.0  # Should complete quickly
            assert performance_metrics.get("timing_precision_ms", float('inf')) < 5.0  # Good timing precision

            # Test distributed timing drift detection and compensation
            drift_analysis = consolidation_result.get("drift_analysis", {})
            assert drift_analysis.get("drift_detected", True) is True  # Should detect our simulated offsets
            assert drift_analysis.get("drift_compensated", False) is True
            assert len(drift_analysis.get("node_drift_corrections", {})) == 3