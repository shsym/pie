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

    def __init__(self, reference_dir: str):
        """Initialize verifier with reference directory."""
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

        self.test_instance = None
        self.verification_results: List[Dict[str, Any]] = []

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

    def generate_verification_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
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
        # Create verifier
        verifier = TensorReferenceVerifier(args.reference_dir)

        # Run verification
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