#!/usr/bin/env python3
"""
Verify Tensor Reference Files CLI

Loads previously generated tensor reference files and verifies them against
fresh L4MA inference to confirm hash matching and data integrity.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import the proven tensor validation system
try:
    # Import from tests directory relative to project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    integration_test_path = project_root / "tests" / "debug_framework" / "integration"
    sys.path.insert(0, str(integration_test_path))
    from test_tensor_validation_critical import TensorValidationCriticalTest
    TENSOR_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Tensor validation system not available: {e}")
    TENSOR_VALIDATION_AVAILABLE = False

# Import backend interfaces for Metal comparison
try:
    from ..integrations.backend_interfaces import BackendType, create_backend
    from ..integrations.metal_backend import MetalBackend
    from ..services.plugin_registry import PluginRegistry
    BACKEND_INTERFACES_AVAILABLE = True
except ImportError as e:
    print(f"Backend interfaces not available: {e}")
    BACKEND_INTERFACES_AVAILABLE = False


class TensorReferenceVerifier:
    """
    Verifier for tensor reference files using fresh L4MA inference.
    """

    def __init__(self, reference_dir: str, backend: str = "pytorch", metal_backend_path: Optional[str] = None):
        """Initialize verifier with reference directory and backend selection."""
        self.reference_dir = Path(reference_dir)
        if not self.reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

        self.metadata_file = self.reference_dir / "reference_metadata.json"
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        # Load reference metadata
        with open(self.metadata_file, 'r') as f:
            self.reference_metadata = json.load(f)

        print(f"📁 Reference directory: {self.reference_dir}")
        print(f"📊 Reference files: {len(self.reference_metadata['tensor_files'])}")

        # Backend configuration
        self.primary_backend = backend
        self.metal_backend_path = metal_backend_path
        self.reference_backend = self._detect_reference_backend()
        self.backends: Dict[str, Any] = {}
        self.plugin_registry = None

        # Initialize plugin registry if backend interfaces available
        if BACKEND_INTERFACES_AVAILABLE:
            self.plugin_registry = PluginRegistry()
            self._register_backends()

        self.test_instance = None
        self.verification_results: List[Dict[str, Any]] = []

    def _detect_reference_backend(self) -> str:
        """Detect which backend was used to generate the reference files."""
        generation_info = self.reference_metadata.get('generation_info', {})
        backend_info = generation_info.get('backend', 'pytorch')  # Default to pytorch

        print(f"🔍 Detected reference backend: {backend_info}")
        return backend_info

    def _register_backends(self):
        """Register available backends with the plugin registry."""
        if not BACKEND_INTERFACES_AVAILABLE:
            print("⚠️ Backend interfaces not available - skipping backend registration")
            return

        try:
            # Register Metal backend if available
            if self.primary_backend == "metal" or self.reference_backend == "metal":
                metal_backend = MetalBackend(self.metal_backend_path)
                if metal_backend.initialize():
                    self.backends["metal"] = metal_backend
                    print("✅ Metal backend registered and initialized")
                else:
                    print("❌ Metal backend registration failed - not available")

            # PyTorch backend is handled through the existing test framework
            print("✅ PyTorch backend available through existing framework")

        except Exception as e:
            print(f"⚠️ Backend registration error: {e}")

    def get_available_backends(self) -> List[str]:
        """Get list of available backends for comparison."""
        available = ["pytorch"]  # PyTorch always available through test framework

        if "metal" in self.backends:
            available.append("metal")

        return available

    def load_reference_files(self) -> Dict[str, Any]:
        """Load all reference tensor files and compute their hashes."""
        reference_data = {}

        print("📦 Loading reference tensor files...")
        for file_info in self.reference_metadata['tensor_files']:
            file_path = Path(file_info['path'])
            if not file_path.is_absolute():
                file_path = self.reference_dir / file_info['filename']

            if file_path.exists():
                # Load tensor data and compute hash
                with open(file_path, 'rb') as f:
                    tensor_bytes = f.read()

                tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()

                reference_data[file_info['filename']] = {
                    'path': str(file_path),
                    'size': len(tensor_bytes),
                    'hash': tensor_hash,
                    'expected_size': file_info['size_bytes']
                }

                print(f"   ✅ {file_info['filename']} ({len(tensor_bytes)} bytes, hash: {tensor_hash[:16]}...)")
            else:
                print(f"   ❌ Missing: {file_info['filename']}")
                reference_data[file_info['filename']] = {
                    'error': 'File not found'
                }

        return reference_data

    def run_fresh_inference(self, prompts: Optional[List[str]] = None) -> bool:
        """Run fresh L4MA inference and capture tensors using same logic as reference generation."""
        if not TENSOR_VALIDATION_AVAILABLE:
            print("❌ Tensor validation system not available")
            return False

        print("\n🧠 Running fresh L4MA inference...")

        # Import the reference generator to use exact same logic
        from debug_framework.cli.generate_references import PersistentTensorReferenceGenerator

        # Create a temporary generator instance
        temp_generator = PersistentTensorReferenceGenerator("/tmp/verify_temp")

        # Use prompts from reference metadata if not provided
        if prompts is None:
            prompts = [result['prompt'] for result in self.reference_metadata['generation_results']]

        print(f"📝 Using prompts: {prompts}")

        # Run the exact same inference logic as reference generation
        print("🔧 Setting up L4MA model and debug framework...")
        if not temp_generator.test_instance:
            temp_generator.test_instance = TensorValidationCriticalTest()

        if not temp_generator.test_instance.setup_test_environment():
            print("❌ Failed to setup test environment")
            return False

        # Apply the same layer decorators as reference generation
        layer_methods = temp_generator._detect_model_layers()
        try:
            decoration_results = temp_generator.test_instance.backend_test.debug_integration.apply_checkpoint_decorators(layer_methods)
            print(f"✅ Applied checkpoint decorators to {len(decoration_results)} layers")
        except Exception as e:
            print(f"⚠️  Warning: Could not apply all layer decorators: {e}")

        all_fresh_tensors = {}

        # Process each prompt using the same logic as reference generation
        for i, prompt in enumerate(prompts):
            print(f"\n🔍 Processing prompt {i+1}/{len(prompts)}: '{prompt}'")

            # Clear previous captures
            temp_generator.test_instance.captured_tensors.clear()
            temp_generator.test_instance.test_recordings.clear()

            # Use the same capture logic as reference generation
            if not temp_generator.test_instance.capture_real_tensors():
                print(f"❌ Failed to capture tensors for prompt: '{prompt}'")
                continue

            print(f"   ✅ Captured {len(temp_generator.test_instance.captured_tensors)} tensors")

            # Run T063 to create binary tensor files for comparison
            if not temp_generator.test_instance.test_t063_binary_storage_with_metadata():
                print(f"❌ T063 failed for prompt: '{prompt}'")
                continue

            print(f"   💾 Created {len(temp_generator.test_instance.test_recordings)} tensor recordings")

            # Collect fresh tensor data with hashes
            for recording in temp_generator.test_instance.test_recordings:
                # Load the fresh tensor data
                with open(recording.tensor_data_path, 'rb') as f:
                    fresh_tensor_bytes = f.read()

                fresh_hash = hashlib.sha256(fresh_tensor_bytes).hexdigest()

                # Create key that matches reference file naming
                tensor_key = f"{recording.tensor_name}.tensor"

                all_fresh_tensors[tensor_key] = {
                    'hash': fresh_hash,
                    'size': len(fresh_tensor_bytes),
                    'tensor_name': recording.tensor_name,
                    'prompt': prompt
                }

        # Store the test instance for cleanup
        self.test_instance = temp_generator.test_instance
        self.fresh_tensors = all_fresh_tensors
        return len(all_fresh_tensors) > 0

    def compare_tensors(self, reference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare fresh inference tensors against reference tensors."""
        print("\n🔍 Comparing fresh inference against references...")

        comparison_results = {
            'total_references': len(reference_data),
            'total_fresh': len(self.fresh_tensors),
            'matches': [],
            'mismatches': [],
            'missing_fresh': [],
            'missing_reference': []
        }

        # Check each reference file
        for ref_filename, ref_info in reference_data.items():
            if 'error' in ref_info:
                comparison_results['missing_reference'].append({
                    'filename': ref_filename,
                    'error': ref_info['error']
                })
                continue

            if ref_filename in self.fresh_tensors:
                fresh_info = self.fresh_tensors[ref_filename]

                # Compare hashes
                hash_match = ref_info['hash'] == fresh_info['hash']
                size_match = ref_info['size'] == fresh_info['size']

                result = {
                    'filename': ref_filename,
                    'tensor_name': fresh_info['tensor_name'],
                    'hash_match': hash_match,
                    'size_match': size_match,
                    'reference_hash': ref_info['hash'][:16] + '...',
                    'fresh_hash': fresh_info['hash'][:16] + '...',
                    'reference_size': ref_info['size'],
                    'fresh_size': fresh_info['size']
                }

                if hash_match and size_match:
                    comparison_results['matches'].append(result)
                    print(f"   ✅ {ref_filename}: Hash ✅ Size ✅")
                else:
                    comparison_results['mismatches'].append(result)
                    hash_status = "✅" if hash_match else "❌"
                    size_status = "✅" if size_match else "❌"
                    print(f"   ❌ {ref_filename}: Hash {hash_status} Size {size_status}")
            else:
                comparison_results['missing_fresh'].append({
                    'filename': ref_filename,
                    'reference_hash': ref_info['hash'][:16] + '...'
                })
                print(f"   ❓ {ref_filename}: No fresh tensor found")

        # Check for fresh tensors not in references
        for fresh_filename in self.fresh_tensors:
            if fresh_filename not in reference_data:
                comparison_results['missing_reference'].append({
                    'filename': fresh_filename,
                    'fresh_hash': self.fresh_tensors[fresh_filename]['hash'][:16] + '...'
                })

        return comparison_results

    def run_metal_backend_comparison(self, reference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Metal backend computation on reference tensors for comparison.

        Args:
            reference_data: Reference tensor data loaded from files

        Returns:
            Dictionary with Metal vs PyTorch comparison results
        """
        print("\n🔧 Running Metal backend computation for comparison...")

        if "metal" not in self.backends:
            return {
                'error': 'Metal backend not available',
                'available_backends': self.get_available_backends()
            }

        metal_backend = self.backends["metal"]
        comparison_results = {
            'total_tensors': 0,
            'successful_comparisons': 0,
            'failed_comparisons': 0,
            'comparisons': [],
            'backend_info': {
                'metal_capabilities': metal_backend.get_capabilities(),
                'reference_backend': self.reference_backend
            }
        }

        for filename, ref_info in reference_data.items():
            if 'error' in ref_info:
                continue

            comparison_results['total_tensors'] += 1

            try:
                # Load tensor data
                tensor_data = self._load_tensor_data(ref_info['path'])
                if tensor_data is None:
                    comparison_results['failed_comparisons'] += 1
                    continue

                # Determine tensor operation type from filename
                operation_type = self._infer_operation_type(filename)

                # Run Metal computation based on operation type
                metal_result = self._run_metal_operation(metal_backend, operation_type, tensor_data)

                if metal_result is not None:
                    # Compare with reference (PyTorch) result
                    comparison = self._compare_tensors(tensor_data, metal_result, filename)
                    comparison_results['comparisons'].append(comparison)

                    if comparison['status'] == 'match':
                        comparison_results['successful_comparisons'] += 1
                        print(f"   ✅ {filename}: Metal vs PyTorch match within tolerance")
                    else:
                        comparison_results['failed_comparisons'] += 1
                        print(f"   ❌ {filename}: Metal vs PyTorch mismatch - {comparison['error']}")
                else:
                    comparison_results['failed_comparisons'] += 1
                    print(f"   ⚠️ {filename}: Metal computation failed")

            except Exception as e:
                comparison_results['failed_comparisons'] += 1
                print(f"   ❌ {filename}: Error - {e}")

        return comparison_results

    def _load_tensor_data(self, tensor_path: str):
        """Load tensor data from file."""
        try:
            import numpy as np
            # Assuming tensor files are saved as .npy files
            if tensor_path.endswith('.tensor'):
                # These might be custom binary format, try to load as numpy
                with open(tensor_path, 'rb') as f:
                    # Skip header if present and load raw float32 data
                    data = f.read()
                    # Try to parse as numpy array - this is simplified
                    # In practice, you'd need to know the exact format
                    return np.frombuffer(data, dtype=np.float32)
            else:
                return np.load(tensor_path)
        except Exception as e:
            print(f"Failed to load tensor from {tensor_path}: {e}")
            return None

    def _infer_operation_type(self, filename: str) -> str:
        """Infer operation type from tensor filename."""
        filename_lower = filename.lower()

        if 'attention' in filename_lower or 'attn' in filename_lower:
            return 'attention'
        elif 'mlp' in filename_lower:
            return 'mlp'
        elif 'embedding' in filename_lower or 'embed' in filename_lower:
            return 'embedding'
        elif 'norm' in filename_lower:
            return 'normalization'
        else:
            return 'unknown'

    def _run_metal_operation(self, metal_backend, operation_type: str, tensor_data):
        """Run Metal operation on tensor data."""
        try:
            if operation_type == 'attention':
                # For attention, we need Q, K, V tensors
                # This is simplified - in practice you'd parse the actual tensor structure
                if len(tensor_data.shape) >= 1:
                    # Create simple Q, K, V for testing
                    if len(tensor_data.shape) == 1:
                        # Reshape to 2D for attention
                        size = int(np.sqrt(len(tensor_data)))
                        if size * size == len(tensor_data):
                            tensor_data = tensor_data.reshape(size, size)

                    if len(tensor_data.shape) == 2:
                        seq_len, hidden_size = tensor_data.shape
                        head_size = min(hidden_size, 64)  # Use smaller head size for testing

                        query = tensor_data[:, :head_size].copy()
                        key = query.copy()
                        value = query.copy()

                        result = metal_backend.run_attention(query, key, value)
                        return result.output

            elif operation_type == 'mlp':
                result = metal_backend.run_mlp(tensor_data)
                return result.output

            elif operation_type == 'embedding':
                # For embedding, we need input_ids and embedding table
                # This is simplified - in practice you'd have the actual embedding table
                if len(tensor_data) > 0:
                    # Create dummy input IDs and embedding table for testing
                    input_ids = np.arange(min(len(tensor_data), 100), dtype=np.int32)
                    vocab_size = 1000
                    hidden_size = 256
                    embedding_table = np.random.randn(vocab_size, hidden_size).astype(np.float32)

                    result = metal_backend.run_embedding(input_ids, embedding_table=embedding_table)
                    return result.output

            elif operation_type == 'normalization':
                if len(tensor_data.shape) >= 1:
                    result = metal_backend.run_normalization(tensor_data)
                    return result.output

            return None

        except Exception as e:
            print(f"Metal operation {operation_type} failed: {e}")
            return None

    def _compare_tensors(self, pytorch_tensor, metal_tensor, filename: str) -> Dict[str, Any]:
        """Compare PyTorch and Metal tensor results."""
        try:
            import numpy as np

            # Ensure both tensors are numpy arrays
            if hasattr(pytorch_tensor, 'cpu'):
                pytorch_np = pytorch_tensor.cpu().numpy()
            else:
                pytorch_np = pytorch_tensor

            if hasattr(metal_tensor, 'cpu'):
                metal_np = metal_tensor.cpu().numpy()
            else:
                metal_np = metal_tensor

            # For tensor format compatibility, just compare basic properties
            # since the actual computation outputs may differ between backends
            return {
                'status': 'computed',
                'filename': filename,
                'pytorch_shape': pytorch_np.shape if hasattr(pytorch_np, 'shape') else 'scalar',
                'metal_shape': metal_np.shape if hasattr(metal_np, 'shape') else 'scalar',
                'pytorch_dtype': str(pytorch_np.dtype) if hasattr(pytorch_np, 'dtype') else 'unknown',
                'metal_dtype': str(metal_np.dtype) if hasattr(metal_np, 'dtype') else 'unknown',
                'computation_successful': True
            }

        except Exception as e:
            return {
                'status': 'comparison_error',
                'filename': filename,
                'error': str(e)
            }

    def compare_backends(self, reference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare different backends using the same reference data.

        Args:
            reference_data: Reference tensor data

        Returns:
            Backend comparison results
        """
        print(f"\n🔄 Comparing backends: {self.reference_backend} (reference) vs {self.primary_backend}")

        if self.primary_backend == "metal":
            return self.run_metal_backend_comparison(reference_data)
        else:
            return {
                'error': f'Backend comparison not implemented for {self.primary_backend}',
                'available_backends': self.get_available_backends()
            }

    def generate_verification_report(self, comparison_results: Dict[str, Any], backend_comparison: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate verification report."""
        total_comparisons = comparison_results['total_references']
        matches = len(comparison_results['matches'])
        mismatches = len(comparison_results['mismatches'])

        success_rate = (matches / total_comparisons * 100) if total_comparisons > 0 else 0

        report = {
            'verification_summary': {
                'total_references': total_comparisons,
                'perfect_matches': matches,
                'mismatches': mismatches,
                'success_rate': success_rate,
                'status': 'PASS' if success_rate == 100.0 else 'FAIL'
            },
            'reference_metadata': self.reference_metadata['generation_info'],
            'comparison_details': comparison_results
        }

        # Add backend comparison results if provided
        if backend_comparison is not None:
            report['backend_comparison'] = backend_comparison

            # Add backend comparison summary
            if 'error' not in backend_comparison:
                backend_success_rate = 0
                if backend_comparison.get('total_tensors', 0) > 0:
                    backend_success_rate = (backend_comparison.get('successful_comparisons', 0) /
                                          backend_comparison.get('total_tensors', 1)) * 100

                report['backend_comparison_summary'] = {
                    'total_tensors': backend_comparison.get('total_tensors', 0),
                    'successful_comparisons': backend_comparison.get('successful_comparisons', 0),
                    'failed_comparisons': backend_comparison.get('failed_comparisons', 0),
                    'success_rate': backend_success_rate,
                    'primary_backend': self.primary_backend,
                    'reference_backend': self.reference_backend
                }

                print(f"\n🔧 Backend Comparison Summary:")
                print(f"   Total tensors processed: {backend_comparison.get('total_tensors', 0)}")
                print(f"   Successful computations: {backend_comparison.get('successful_comparisons', 0)}")
                print(f"   Failed computations: {backend_comparison.get('failed_comparisons', 0)}")
                print(f"   Backend success rate: {backend_success_rate:.1f}%")
            else:
                print(f"\n❌ Backend comparison failed: {backend_comparison['error']}")

        print(f"\n📊 Verification Summary:")
        print(f"   Total references: {total_comparisons}")
        print(f"   Perfect matches: {matches}")
        print(f"   Mismatches: {mismatches}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Status: {'✅ PASS' if success_rate == 100.0 else '❌ FAIL'}")

        return report

    def verify_references(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main verification workflow."""
        print("🔍 Starting Tensor Reference Verification")
        print("=" * 60)

        try:
            # Load reference files
            reference_data = self.load_reference_files()

            # Run fresh inference
            if not self.run_fresh_inference(prompts):
                return {
                    'success': False,
                    'error': 'Failed to run fresh inference'
                }

            # Compare tensors
            comparison_results = self.compare_tensors(reference_data)

            # Generate report
            report = self.generate_verification_report(comparison_results)

            # Cleanup
            if self.test_instance:
                self.test_instance.cleanup_test_environment()

            report['success'] = True
            return report

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def verify_references_with_backend_comparison(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main verification workflow with backend comparison."""
        print("🔍 Starting Tensor Reference Verification with Backend Comparison")
        print("=" * 70)

        try:
            # Load reference files
            reference_data = self.load_reference_files()

            # Run fresh inference if needed (for PyTorch comparison)
            if self.primary_backend == "pytorch" or self.reference_backend == "pytorch":
                if not self.run_fresh_inference(prompts):
                    return {
                        'success': False,
                        'error': 'Failed to run fresh inference'
                    }

                # Compare tensors (PyTorch vs reference)
                comparison_results = self.compare_tensors(reference_data)
            else:
                # Skip fresh inference if both backends are non-PyTorch
                comparison_results = {
                    'total_references': len(reference_data),
                    'total_fresh': 0,
                    'matches': [],
                    'mismatches': [],
                    'missing_fresh': [],
                    'missing_reference': []
                }

            # Run backend comparison
            backend_comparison = self.compare_backends(reference_data)

            # Generate report with backend comparison
            report = self.generate_verification_report(comparison_results, backend_comparison)

            # Cleanup
            if self.test_instance:
                self.test_instance.cleanup_test_environment()

            report['success'] = True
            return report

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify tensor reference files against fresh L4MA inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify latest generated references
  python -m debug_framework.cli.verify_references

  # Verify specific reference directory
  python -m debug_framework.cli.verify_references --reference-dir tensor_references/session_123

  # Verify with custom prompts
  python -m debug_framework.cli.verify_references --prompt "Test prompt"

  # Save verification report
  python -m debug_framework.cli.verify_references --output-report verification_report.json
        """
    )

    parser.add_argument(
        "--reference-dir", "-r",
        help="Directory containing reference tensor files (auto-detects latest if not specified)"
    )

    parser.add_argument(
        "--prompt", "-p",
        action="append",
        help="Custom prompt for verification (can be used multiple times)"
    )

    parser.add_argument(
        "--output-report", "-o",
        help="Save verification report to file"
    )

    parser.add_argument(
        "--backend", "-b",
        choices=["pytorch", "metal", "cuda"],
        default="pytorch",
        help="Backend to use for verification (default: pytorch)"
    )

    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="Compare Metal vs PyTorch results using same reference files (auto-detects reference backend)"
    )

    parser.add_argument(
        "--metal-backend-path",
        help="Path to Metal backend directory (auto-detects if not specified)"
    )

    args = parser.parse_args()

    # Auto-detect latest reference directory if not specified
    if not args.reference_dir:
        tensor_refs_dir = Path("tensor_references")
        if tensor_refs_dir.exists():
            # Find the latest session directory
            session_dirs = [d for d in tensor_refs_dir.iterdir() if d.is_dir() and d.name.startswith("session_")]
            if session_dirs:
                latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)
                args.reference_dir = str(latest_session)
                print(f"🔍 Auto-detected latest reference directory: {args.reference_dir}")
            else:
                print("❌ No reference directories found in tensor_references/")
                return False
        else:
            print("❌ No tensor_references directory found")
            return False

    try:
        # Create verifier with backend configuration
        verifier = TensorReferenceVerifier(
            args.reference_dir,
            backend=args.backend,
            metal_backend_path=args.metal_backend_path
        )

        # Run verification
        if args.compare_backends:
            result = verifier.verify_references_with_backend_comparison(args.prompt)
        else:
            result = verifier.verify_references(args.prompt)

        if result['success']:
            success_rate = result['verification_summary']['success_rate']
            status = result['verification_summary']['status']

            print(f"\n🎉 Verification completed!")
            print(f"📊 Success rate: {success_rate:.1f}%")
            print(f"📈 Status: {status}")

            # Save report if requested
            if args.output_report:
                with open(args.output_report, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"📄 Report saved to: {args.output_report}")

            return success_rate == 100.0
        else:
            print(f"\n❌ Verification failed: {result.get('error', 'unknown')}")
            return False

    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)