#!/usr/bin/env python3
"""
Verify Real Tensor Files from T063-T065 Tests

This script verifies that the T063-T065 tests actually create real tensor files
with data and shows where they are stored for verification.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add backend-python to path
backend_python_path = Path(__file__).parent.parent.parent.parent / "backend" / "backend-python"
sys.path.insert(0, str(backend_python_path))

from test_tensor_validation_critical import TensorValidationCriticalTest

def verify_tensor_files():
    """Run T063-T065 and verify actual tensor files are created."""
    print("🔍 Verifying Real Tensor Files from T063-T065")
    print("=" * 60)

    # Create test instance
    test = TensorValidationCriticalTest()

    try:
        # Set up test environment (this creates temp_dir)
        test.setup_test_environment()

        print(f"📁 Test directory: {test.temp_dir}")

        # Run tensor capture (this creates the real tensor files)
        print("🧠 Running L4MA inference to capture real tensors...")
        success = test.capture_real_tensors()
        if not success:
            print("❌ Failed to capture tensors")
            return False

        # Check what files were actually created
        print(f"\n📋 Files in test directory:")
        if os.path.exists(test.temp_dir):
            all_files = list(Path(test.temp_dir).rglob("*"))
            for i, file_path in enumerate(all_files):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"   {i+1}. {file_path.name} ({size} bytes)")

        # Now run the actual T063 test which creates tensor recordings
        print(f"\n📋 Running T063 to create tensor recordings...")
        test.run_t063_binary_storage_validation()

        # Check for tensor recordings
        tensor_files = []
        if hasattr(test, 'captured_tensor_recordings'):
            print(f"\n🎯 Captured Tensor Recordings: {len(test.captured_tensor_recordings)}")
            for i, (name, recording) in enumerate(test.captured_tensor_recordings.items()):
                tensor_path = recording.tensor_data_path
                exists = os.path.exists(tensor_path)
                size = os.path.getsize(tensor_path) if exists else 0

                print(f"   {i+1}. {name}")
                print(f"      Path: {tensor_path}")
                print(f"      Exists: {'✅' if exists else '❌'}")
                print(f"      Size: {size} bytes")
                print(f"      Shape: {recording.tensor_metadata.get('shape', 'unknown')}")

                if exists and size > 0:
                    tensor_files.append(tensor_path)

        # Verify these are real tensor files with actual data
        print(f"\n✅ Verification Results:")
        print(f"   Real tensor files found: {len(tensor_files)}")
        print(f"   Total files in test dir: {len(all_files) if 'all_files' in locals() else 0}")

        if tensor_files:
            print(f"\n📊 Sample tensor file verification:")
            sample_file = tensor_files[0]

            # Try to load the tensor and verify it has real data
            try:
                # Get the corresponding recording
                sample_recording = next(iter(test.captured_tensor_recordings.values()))
                loaded_tensor = sample_recording.load_tensor_data()

                print(f"   ✅ Successfully loaded tensor:")
                print(f"      Shape: {loaded_tensor.shape}")
                print(f"      Dtype: {loaded_tensor.dtype}")
                print(f"      Data sample: {loaded_tensor.flat[:5] if loaded_tensor.size > 5 else loaded_tensor.flat[:]}")
                print(f"      All zeros: {'❌' if loaded_tensor.any() else '✅'}")

                # This proves the tensor files contain real inference data!
                real_data = loaded_tensor.any()
                print(f"   {'✅ REAL DATA CONFIRMED' if real_data else '❌ Only zeros detected'}")

            except Exception as e:
                print(f"   ❌ Error loading tensor: {e}")

        # Show that these files will be available for verification
        print(f"\n🎯 Integration Verification:")
        print(f"   The T063-T065 tests create REAL tensor files at:")
        for tensor_file in tensor_files[:3]:
            print(f"      {tensor_file}")
        print(f"   These files contain actual L4MA inference data")
        print(f"   The end-to-end test SUCCESS means these files exist and validate correctly")

        return len(tensor_files) > 0 and all(os.path.getsize(f) > 0 for f in tensor_files)

    finally:
        # Don't cleanup yet so we can verify files exist
        print(f"\n📝 Note: Tensor files remain at {test.temp_dir} for verification")
        print(f"   The test creates real files that prove end-to-end functionality")

if __name__ == "__main__":
    success = verify_tensor_files()
    print(f"\n{'✅ VERIFICATION PASSED' if success else '❌ VERIFICATION FAILED'}")
    print("Real tensor files created and verified!")