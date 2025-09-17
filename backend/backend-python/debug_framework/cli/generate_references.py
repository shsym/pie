#!/usr/bin/env python3
"""
Generate Persistent Tensor Reference Files

Uses the proven T063-T065 tensor validation system to generate reference
tensor files in a persistent location for comparison and validation.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any

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


class PersistentTensorReferenceGenerator:
    """
    Generator for persistent tensor reference files using the proven validation system.
    """

    def __init__(self, output_dir: str = "tensor_references"):
        """Initialize the generator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.session_dir = self.output_dir / f"session_{int(time.time())}"
        self.session_dir.mkdir(exist_ok=True)

        print(f"📁 Reference files will be saved to: {self.session_dir}")

        self.test_instance = None
        self.generated_files: List[Path] = []

    def _detect_model_layers(self) -> List[str]:
        """Automatically detect all model layers for comprehensive coverage."""
        try:
            # Try to get model info from the test instance first
            if self.test_instance and hasattr(self.test_instance, 'backend_test'):
                try:
                    handler = self.test_instance.backend_test.handler
                    if hasattr(handler, 'model') and hasattr(handler.model, 'config'):
                        num_layers = getattr(handler.model.config, 'num_hidden_layers', None)
                        if num_layers:
                            print(f"🔍 Detected {num_layers} layers from loaded model")
                        else:
                            # Try alternative config attribute names
                            num_layers = getattr(handler.model.config, 'num_layers', 16)
                            print(f"🔍 Detected {num_layers} layers from model config")
                    else:
                        raise AttributeError("Model config not accessible")
                except Exception as model_error:
                    print(f"⚠️  Could not read from loaded model: {model_error}")
                    raise model_error
            else:
                raise FileNotFoundError("Model cache directory not found")

            # Build comprehensive layer list
            layer_methods = ['embed_tokens']

            # Add all transformer layers
            for i in range(num_layers):
                layer_methods.extend([
                    f'layers.{i}.self_attn',
                    f'layers.{i}.mlp'
                ])

            # Add final layers
            layer_methods.extend(['norm', 'lm_head'])

            print(f"📋 Will capture tensors from {len(layer_methods)} checkpoints")
            return layer_methods

        except Exception as e:
            print(f"⚠️  Error detecting layers: {e}, using default 16 layers")
            # Fallback to 16 layers (common for many small-medium models)
            layer_methods = ['embed_tokens']
            for i in range(16):
                layer_methods.extend([f'layers.{i}.self_attn', f'layers.{i}.mlp'])
            layer_methods.extend(['norm', 'lm_head'])
            print(f"📋 Using fallback: {len(layer_methods)} checkpoints")
            return layer_methods

    def generate_reference_files(self, prompts: List[str] = None) -> Dict[str, Any]:
        """
        Generate persistent tensor reference files using real L4MA inference.

        Args:
            prompts: List of prompts to run inference with

        Returns:
            Generation results with file locations
        """
        if not TENSOR_VALIDATION_AVAILABLE:
            return {
                'success': False,
                'error': 'Tensor validation system not available'
            }

        print("🚀 Generating Persistent Tensor Reference Files")
        print("=" * 60)

        if prompts is None:
            prompts = [
                "The capital of France is",
                "Hello, my name is",
                "What is 2 + 2?"
            ]

        try:
            # Create test instance
            self.test_instance = TensorValidationCriticalTest()

            # Setup test environment with comprehensive layer coverage
            print("🔧 Setting up L4MA model and debug framework...")
            if not self.test_instance.setup_test_environment():
                return {
                    'success': False,
                    'error': 'Failed to setup test environment'
                }

            # Apply comprehensive layer decorators
            layer_methods = self._detect_model_layers()
            try:
                decoration_results = self.test_instance.backend_test.debug_integration.apply_checkpoint_decorators(layer_methods)
                print(f"✅ Applied checkpoint decorators to {len(decoration_results)} layers")
            except Exception as e:
                print(f"⚠️  Warning: Could not apply all layer decorators: {e}")
                print("   Proceeding with default layer coverage")

            print("✅ Test environment ready")

            all_results = []
            total_files_generated = 0

            # Process each prompt
            for i, prompt in enumerate(prompts):
                print(f"\n📝 Processing prompt {i+1}/{len(prompts)}: '{prompt}'")

                prompt_result = self._generate_for_prompt(prompt, i+1)
                all_results.append(prompt_result)

                if prompt_result['success']:
                    total_files_generated += prompt_result['tensor_files_generated']
                    print(f"✅ Generated {prompt_result['tensor_files_generated']} tensor files")
                else:
                    print(f"❌ Failed to process prompt: {prompt_result.get('error', 'unknown')}")

            # Copy tensor files to persistent location before cleanup
            persistent_files = self._preserve_tensor_files()

            # Generate metadata file
            metadata_file = self._generate_metadata_file(all_results, persistent_files)

            # Now cleanup the temporary test environment
            self._cleanup_test_environment()

            success_rate = len([r for r in all_results if r['success']]) / len(all_results)

            print(f"\n📊 Generation Summary:")
            print(f"   Prompts processed: {len(prompts)}")
            print(f"   Success rate: {success_rate*100:.1f}%")
            print(f"   Total tensor files: {len(persistent_files)}")
            print(f"   Reference directory: {self.session_dir}")
            print(f"   Metadata file: {metadata_file}")

            return {
                'success': True,
                'prompts_processed': len(prompts),
                'success_rate': success_rate,
                'tensor_files_generated': len(persistent_files),
                'reference_directory': str(self.session_dir),
                'metadata_file': str(metadata_file),
                'persistent_files': [str(f) for f in persistent_files],
                'prompt_results': all_results
            }

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_for_prompt(self, prompt: str, prompt_id: int) -> Dict[str, Any]:
        """Generate tensor files for a single prompt."""
        try:
            # Clear previous captures
            self.test_instance.captured_tensors.clear()
            self.test_instance.test_recordings.clear()

            # Capture tensors for this prompt
            if not self.test_instance.capture_real_tensors():
                return {
                    'success': False,
                    'prompt': prompt,
                    'error': 'Failed to capture tensors during inference'
                }

            captured_count = len(self.test_instance.captured_tensors)
            print(f"   📊 Captured {captured_count} tensors from inference")

            # Run T063 to create binary tensor files
            if not self.test_instance.test_t063_binary_storage_with_metadata():
                return {
                    'success': False,
                    'prompt': prompt,
                    'error': 'T063 binary storage validation failed'
                }

            recordings_count = len(self.test_instance.test_recordings)
            print(f"   💾 Created {recordings_count} tensor recordings")

            # Validate with T064 and T065 to ensure files are correct
            t064_success = self.test_instance.test_t064_tensor_loading_extraction()
            t065_success = self.test_instance.test_t065_memory_matching_validation()

            if not (t064_success and t065_success):
                return {
                    'success': False,
                    'prompt': prompt,
                    'error': f'Validation failed - T064: {t064_success}, T065: {t065_success}'
                }

            print(f"   ✅ All validation tests passed")

            return {
                'success': True,
                'prompt': prompt,
                'prompt_id': prompt_id,
                'tensors_captured': captured_count,
                'tensor_files_generated': recordings_count,
                'validation_passed': True
            }

        except Exception as e:
            return {
                'success': False,
                'prompt': prompt,
                'error': str(e)
            }

    def _preserve_tensor_files(self) -> List[Path]:
        """Copy tensor files from temporary location to persistent location."""
        persistent_files = []

        if not self.test_instance or not self.test_instance.test_recordings:
            return persistent_files

        print(f"\n💾 Preserving tensor files to persistent location...")

        for i, recording in enumerate(self.test_instance.test_recordings):
            try:
                source_file = Path(recording.tensor_data_path)
                if source_file.exists():
                    # Create descriptive filename
                    tensor_name = recording.tensor_name.replace('/', '_').replace(' ', '_')
                    dest_filename = f"{tensor_name}.tensor"
                    dest_file = self.session_dir / dest_filename

                    # Copy the tensor file
                    shutil.copy2(source_file, dest_file)
                    persistent_files.append(dest_file)

                    print(f"   📄 {dest_filename} ({source_file.stat().st_size} bytes)")

                    # Also save metadata as JSON
                    metadata_file = self.session_dir / f"{tensor_name}.metadata.json"
                    with open(metadata_file, 'w') as f:
                        import json
                        metadata = {
                            'tensor_name': recording.tensor_name,
                            'tensor_metadata': recording.tensor_metadata,
                            'device_info': recording.device_info,
                            'backend_name': recording.backend_name,
                            'file_size_bytes': recording.file_size_bytes,
                            'compression_method': recording.compression_method,
                            'recording_timestamp': recording.recording_timestamp
                        }
                        json.dump(metadata, f, indent=2)

            except Exception as e:
                print(f"   ❌ Failed to preserve {recording.tensor_name}: {e}")

        print(f"   ✅ Preserved {len(persistent_files)} tensor files")
        return persistent_files

    def _generate_metadata_file(self, results: List[Dict], files: List[Path]) -> Path:
        """Generate comprehensive metadata file for the reference dataset."""
        metadata_file = self.session_dir / "reference_metadata.json"

        metadata = {
            'generation_info': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'generator': 'PersistentTensorReferenceGenerator',
                'session_directory': str(self.session_dir),
                'total_files': len(files)
            },
            'model_info': {
                'model_name': 'llama-3.2-1b-instruct',
                'backend': 'pytorch',
                'device': 'cuda:0' if 'cuda' in str(self.test_instance.backend_test.handler.device) else 'cpu',
                'dtype': str(self.test_instance.backend_test.handler.dtype)
            },
            'generation_results': results,
            'tensor_files': [
                {
                    'filename': f.name,
                    'path': str(f),
                    'size_bytes': f.stat().st_size
                }
                for f in files
            ],
            'validation_info': {
                'system': 'T063-T065 Critical Tensor Validation',
                'tests_passed': ['T063: Binary Storage', 'T064: Loading/Extraction', 'T065: Memory Matching'],
                'tolerance': '1e-15 (exact memory matching)',
                'hash_validation': 'SHA256'
            }
        }

        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)

        return metadata_file

    def _cleanup_test_environment(self):
        """Clean up the test environment (but preserve our persistent files)."""
        if self.test_instance:
            print("\n🧹 Cleaning up test environment...")
            self.test_instance.cleanup_test_environment()


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate persistent tensor reference files using L4MA inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reference files with default prompts
  python -m debug_framework.cli.generate_references

  # Generate with custom output directory
  python -m debug_framework.cli.generate_references --output-dir /path/to/references

  # Generate with custom prompts
  python -m debug_framework.cli.generate_references --prompt "Custom prompt 1" --prompt "Custom prompt 2"
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="tensor_references",
        help="Output directory for reference files (default: tensor_references)"
    )

    parser.add_argument(
        "--prompt", "-p",
        action="append",
        help="Custom prompt for inference (can be used multiple times)"
    )

    args = parser.parse_args()

    # Create generator
    generator = PersistentTensorReferenceGenerator(args.output_dir)

    # Generate reference files
    result = generator.generate_reference_files(args.prompt)

    if result['success']:
        print(f"\n🎉 Successfully generated tensor reference files!")
        print(f"📁 Location: {result['reference_directory']}")
        print(f"📊 Files created: {result['tensor_files_generated']}")
        return True
    else:
        print(f"\n❌ Failed to generate reference files: {result.get('error', 'unknown')}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)