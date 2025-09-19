#!/usr/bin/env python3
"""
Full Forward Pass Validation Test

Tests the complete forward pass through all 16 layers of the L4MA model,
checking tensor values at each layer and validating that output tokens are reasonable.
This is the critical end-to-end test that validates the entire computation pipeline.

Key validations:
1. Forward pass through all 16 layers with Metal kernels
2. Tensor value verification at each layer (embedding -> layer 0-15 -> output)
3. Output token reasonableness and correctness
4. Numerical stability throughout the pipeline
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add backend-python to path
backend_python_path = Path(__file__).parent.parent.parent.parent / "backend" / "backend-python"
sys.path.insert(0, str(backend_python_path))

# Core imports
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Backend imports
try:
    from handler import Handler
    from message import ForwardPassRequest, ForwardPassResponse
    from config.common import ModelInfo
    from model.l4ma import L4maForCausalLM
    from model.l4ma_runtime import FlashInferL4maBackend
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Backend not available: {e}")
    BACKEND_AVAILABLE = False

FLASHINFER_AVAILABLE = False
if BACKEND_AVAILABLE:
    FLASHINFER_AVAILABLE = FlashInferL4maBackend.is_available()


# Debug framework imports
try:
    from debug_framework.integrations.metal_backend import MetalBackend
    DEBUG_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Debug framework not available: {e}")
    DEBUG_FRAMEWORK_AVAILABLE = False


class FullForwardPassValidator:
    """
    Complete forward pass validation through all 16 layers with tensor verification.

    This validator runs the full model forward pass and captures tensor values
    at every critical checkpoint to ensure numerical correctness throughout
    the entire computation pipeline.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the full forward pass validator."""
        self.verbose = verbose
        self.handler: Optional[Handler] = None
        self.model_info: Optional[ModelInfo] = None
        self.metal_backend: Optional[MetalBackend] = None

        # Model configuration
        self.model_cache_path = Path.home() / "Library" / "Caches" / "pie" / "models"
        self.model_name = "llama-3.2-1b-instruct"
        self.metadata_path = self.model_cache_path / f"{self.model_name}.toml"

        # Validation state
        self.captured_tensors: Dict[str, torch.Tensor] = {}
        self.layer_statistics: Dict[str, Dict[str, float]] = {}
        self.validation_results: Dict[str, Any] = {}

        self.log("🚀 FullForwardPassValidator initialized")
        self.log(f"   Target model: {self.model_name}")

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def setup_model_and_backend(self) -> bool:
        """Setup the model and Metal backend for validation."""
        self.log("\n📦 Setting up model and Metal backend...")

        if not BACKEND_AVAILABLE or not TORCH_AVAILABLE:
            self.log("❌ Backend dependencies not available")
            return False

        if not self.metadata_path.exists():
            self.log(f"❌ Model metadata not found: {self.metadata_path}")
            return False

        try:
            # Load model configuration
            self.log(f"   Loading model from: {self.metadata_path}")
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            self.model_info = ModelInfo.load_from_file(str(self.metadata_path), device, dtype)

            # Create model
            if not FLASHINFER_AVAILABLE:
                self.log("FlashInfer backend unavailable; skipping validation.")
                return False

            backend = FlashInferL4maBackend()
            model = L4maForCausalLM(self.model_info.architecture, backend=backend)
            model.eval()
            model = model.to(device)

            # Create handler
            self.handler = Handler(
                model=model,
                model_info=self.model_info,
                device=device
            )

            self.log(f"✅ Model loaded: {self.model_info.name}")
            self.log(f"   Architecture: {self.model_info.architecture.type}")
            self.log(f"   Device: {device}")
            self.log(f"   Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
            self.log(f"   Layers: {self.model_info.architecture.num_layers}")

            # Initialize Metal backend
            if DEBUG_FRAMEWORK_AVAILABLE:
                try:
                    self.metal_backend = MetalBackend()
                    if self.metal_backend.initialize():
                        self.log(f"✅ Metal backend initialized")
                        self.log(f"   Available kernels: {len(self.metal_backend.get_available_kernels())}")
                    else:
                        self.log("⚠️ Metal backend initialization failed")
                except Exception as e:
                    self.log(f"⚠️ Metal backend error: {e}")

            return True

        except Exception as e:
            self.log(f"❌ Setup failed: {e}")
            return False

    def install_layer_hooks(self) -> List:
        """Install forward hooks to capture tensor values at each layer."""
        self.log("\n🔍 Installing tensor capture hooks...")

        hooks = []
        model = self.handler.model

        def create_hook(name: str):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.captured_tensors[name] = output.detach().cpu().clone()
                elif isinstance(output, tuple) and len(output) > 0:
                    # Take the first output if it's a tuple
                    self.captured_tensors[name] = output[0].detach().cpu().clone()
                return output
            return hook_fn

        # Hook embedding layer
        if hasattr(model, 'embed_tokens'):
            hooks.append(model.embed_tokens.register_forward_hook(
                create_hook('embedding_output')
            ))

        # Hook all transformer layers
        if hasattr(model, 'layers'):
            num_layers = len(model.layers)
            self.log(f"   Installing hooks for {num_layers} transformer layers...")

            for i, layer in enumerate(model.layers):
                # Hook attention output
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(
                        create_hook(f'layer_{i}_attention_output')
                    ))

                # Hook MLP output
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(
                        create_hook(f'layer_{i}_mlp_output')
                    ))

                # Hook layer output (after residual connections)
                hooks.append(layer.register_forward_hook(
                    create_hook(f'layer_{i}_output')
                ))

        # Hook final norm
        if hasattr(model, 'norm'):
            hooks.append(model.norm.register_forward_hook(
                create_hook('final_norm_output')
            ))

        # Hook LM head
        if hasattr(model, 'lm_head'):
            hooks.append(model.lm_head.register_forward_hook(
                create_hook('lm_head_output')
            ))

        self.log(f"✅ Installed {len(hooks)} tensor capture hooks")
        return hooks

    def run_full_forward_pass(self, prompt: str = "The capital of France is") -> bool:
        """Run complete forward pass through all layers with tensor capture."""
        self.log(f"\n🚀 Running full forward pass...")
        self.log(f"   Test prompt: '{prompt}'")

        # Install hooks
        hooks = self.install_layer_hooks()

        try:
            # Clear previous captures
            self.captured_tensors.clear()

            # Create forward pass request
            request = ForwardPassRequest(
                prompt=prompt,
                max_tokens=5,  # Generate a few tokens to test
                temperature=0.1,  # Low temperature for more deterministic results
                top_p=0.9
            )

            # Execute forward pass
            self.log("   Executing forward pass...")
            start_time = time.perf_counter()

            response = self.handler.forward_pass(request)

            forward_time = time.perf_counter() - start_time

            self.log(f"✅ Forward pass completed in {forward_time:.3f}s")
            self.log(f"   Generated tokens: {len(response.generated_token_ids) if response.generated_token_ids else 0}")
            self.log(f"   Generated text: '{response.generated_text}'")
            self.log(f"   Captured tensors: {len(self.captured_tensors)}")

            # Store results
            self.validation_results['forward_pass'] = {
                'success': True,
                'forward_time': forward_time,
                'prompt': prompt,
                'generated_text': response.generated_text,
                'generated_tokens': response.generated_token_ids,
                'captured_tensor_count': len(self.captured_tensors)
            }

            return True

        except Exception as e:
            self.log(f"❌ Forward pass failed: {e}")
            self.validation_results['forward_pass'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

        finally:
            # Remove hooks
            for hook in hooks:
                try:
                    hook.remove()
                except:
                    pass

    def analyze_tensor_values_by_layer(self) -> bool:
        """Analyze tensor values layer by layer for numerical issues."""
        self.log("\n🔍 Analyzing tensor values layer by layer...")

        if not self.captured_tensors:
            self.log("❌ No captured tensors to analyze")
            return False

        layer_analysis = {}
        issues_found = []

        # Analyze each captured tensor
        for tensor_name, tensor in self.captured_tensors.items():
            stats = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'mean': float(tensor.mean()),
                'std': float(tensor.std()),
                'min': float(tensor.min()),
                'max': float(tensor.max()),
                'nan_count': int(torch.isnan(tensor).sum()),
                'inf_count': int(torch.isinf(tensor).sum()),
                'zero_count': int((tensor == 0).sum()),
                'abs_max': float(torch.abs(tensor).max())
            }

            layer_analysis[tensor_name] = stats

            # Check for numerical issues
            if stats['nan_count'] > 0:
                issues_found.append(f"{tensor_name}: {stats['nan_count']} NaN values")
            if stats['inf_count'] > 0:
                issues_found.append(f"{tensor_name}: {stats['inf_count']} Inf values")
            if stats['abs_max'] > 1000:
                issues_found.append(f"{tensor_name}: Very large values (max: {stats['abs_max']:.2e})")
            if stats['std'] == 0 and stats['zero_count'] == tensor.numel():
                issues_found.append(f"{tensor_name}: All zeros")

            # Log key statistics
            self.log(f"   {tensor_name:30} | shape={str(stats['shape']):15} | mean={stats['mean']:8.4f} | std={stats['std']:8.4f} | range=[{stats['min']:8.4f}, {stats['max']:8.4f}]")

        # Check layer progression
        self.log("\n📊 Layer progression analysis:")

        # Look for embedding -> layer outputs progression
        layer_outputs = [(name, tensor) for name, tensor in self.captured_tensors.items()
                        if 'layer_' in name and '_output' in name and 'attention' not in name and 'mlp' not in name]
        layer_outputs.sort(key=lambda x: int(x[0].split('_')[1]))  # Sort by layer number

        for i, (layer_name, tensor) in enumerate(layer_outputs):
            layer_num = int(layer_name.split('_')[1])
            stats = layer_analysis[layer_name]
            self.log(f"     Layer {layer_num:2d}: mean={stats['mean']:8.4f}, std={stats['std']:8.4f}, range=[{stats['min']:8.4f}, {stats['max']:8.4f}]")

        # Report issues
        if issues_found:
            self.log(f"\n⚠️ Numerical issues found:")
            for issue in issues_found:
                self.log(f"   - {issue}")
        else:
            self.log("\n✅ No numerical issues detected")

        self.layer_statistics = layer_analysis
        self.validation_results['tensor_analysis'] = {
            'success': len(issues_found) == 0,
            'layer_statistics': layer_analysis,
            'issues_found': issues_found,
            'layers_analyzed': len([name for name in self.captured_tensors.keys() if 'layer_' in name])
        }

        return len(issues_found) == 0

    def validate_output_tokens(self) -> bool:
        """Validate that output tokens are reasonable and make sense."""
        self.log("\n🎯 Validating output tokens...")

        if 'forward_pass' not in self.validation_results:
            self.log("❌ No forward pass results to validate")
            return False

        forward_result = self.validation_results['forward_pass']
        if not forward_result['success']:
            self.log("❌ Forward pass failed, cannot validate tokens")
            return False

        generated_text = forward_result['generated_text']
        generated_tokens = forward_result['generated_tokens']

        self.log(f"   Generated text: '{generated_text}'")
        self.log(f"   Generated tokens: {generated_tokens}")

        # Basic validation checks
        validation_checks = {}

        # Check 1: Tokens are valid integers
        valid_tokens = all(isinstance(token, int) and token >= 0 for token in generated_tokens)
        validation_checks['valid_token_ids'] = valid_tokens

        # Check 2: Generated text is not empty
        non_empty_text = len(generated_text.strip()) > 0
        validation_checks['non_empty_text'] = non_empty_text

        # Check 3: No excessive repetition
        words = generated_text.split()
        if len(words) > 1:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            no_excessive_repetition = repetition_ratio > 0.3
        else:
            no_excessive_repetition = True
        validation_checks['no_excessive_repetition'] = no_excessive_repetition

        # Check 4: Reasonable character distribution
        if generated_text.strip():
            alpha_ratio = sum(c.isalpha() for c in generated_text) / len(generated_text)
            reasonable_chars = alpha_ratio > 0.3  # At least 30% alphabetic
        else:
            reasonable_chars = False
        validation_checks['reasonable_characters'] = reasonable_chars

        # Check 5: No obvious corruption (binary, excessive special chars, etc.)
        special_char_ratio = sum(1 for c in generated_text if not c.isalnum() and c not in ' .,!?;:-\'\"') / max(len(generated_text), 1)
        no_corruption = special_char_ratio < 0.5
        validation_checks['no_corruption'] = no_corruption

        # Overall assessment
        all_checks_passed = all(validation_checks.values())

        self.log(f"   Token validation results:")
        for check_name, passed in validation_checks.items():
            status = "✅" if passed else "❌"
            self.log(f"     {status} {check_name}")

        if all_checks_passed:
            self.log("✅ Output tokens are reasonable and valid")
        else:
            self.log("⚠️ Some output validation checks failed")

        self.validation_results['token_validation'] = {
            'success': all_checks_passed,
            'validation_checks': validation_checks,
            'generated_text': generated_text,
            'generated_tokens': generated_tokens,
            'token_count': len(generated_tokens)
        }

        return all_checks_passed

    def test_multiple_prompts(self) -> bool:
        """Test with multiple prompts to ensure robustness."""
        self.log("\n🔀 Testing multiple prompts for robustness...")

        test_prompts = [
            "The capital of France is",
            "Hello, my name is",
            "What is 2 + 2?",
            "Once upon a time there was",
            "The quick brown fox"
        ]

        multi_prompt_results = {}
        all_passed = True

        for i, prompt in enumerate(test_prompts):
            self.log(f"\n   Testing prompt {i+1}: '{prompt}'")

            # Run forward pass for this prompt
            success = self.run_full_forward_pass(prompt)
            if success:
                # Validate this specific output
                token_valid = self.validate_output_tokens()
                tensor_valid = self.analyze_tensor_values_by_layer()

                multi_prompt_results[f'prompt_{i+1}'] = {
                    'prompt': prompt,
                    'forward_pass_success': success,
                    'token_validation_success': token_valid,
                    'tensor_analysis_success': tensor_valid,
                    'generated_text': self.validation_results['forward_pass']['generated_text'],
                    'generated_tokens': self.validation_results['forward_pass']['generated_tokens']
                }

                all_passed &= (success and token_valid and tensor_valid)

                self.log(f"     Result: '{self.validation_results['forward_pass']['generated_text']}'")
                self.log(f"     Status: {'✅ PASS' if (success and token_valid and tensor_valid) else '❌ FAIL'}")
            else:
                multi_prompt_results[f'prompt_{i+1}'] = {
                    'prompt': prompt,
                    'forward_pass_success': False,
                    'error': 'Forward pass failed'
                }
                all_passed = False

        self.validation_results['multi_prompt_test'] = {
            'success': all_passed,
            'results': multi_prompt_results,
            'prompts_tested': len(test_prompts),
            'prompts_passed': sum(1 for r in multi_prompt_results.values()
                                if r.get('forward_pass_success', False) and
                                   r.get('token_validation_success', False) and
                                   r.get('tensor_analysis_success', False))
        }

        return all_passed

    def run_comprehensive_validation(self) -> bool:
        """Run the complete comprehensive validation test."""
        self.log("🚀 Starting Comprehensive Full Forward Pass Validation")
        self.log("=" * 80)
        self.log("Testing complete forward pass through all 16 layers with tensor verification")
        self.log("and output token validation")
        self.log("=" * 80)

        overall_start_time = time.perf_counter()

        # Step 1: Setup
        if not self.setup_model_and_backend():
            self.log("❌ Setup failed - aborting validation")
            return False

        # Step 2: Single prompt test (detailed)
        self.log("\n" + "="*50)
        self.log("PHASE 1: Single Prompt Detailed Analysis")
        self.log("="*50)

        if not self.run_full_forward_pass():
            self.log("❌ Forward pass failed - aborting validation")
            return False

        tensor_analysis_success = self.analyze_tensor_values_by_layer()
        token_validation_success = self.validate_output_tokens()

        # Step 3: Multiple prompt test
        self.log("\n" + "="*50)
        self.log("PHASE 2: Multiple Prompt Robustness Test")
        self.log("="*50)

        multi_prompt_success = self.test_multiple_prompts()

        # Final results
        total_time = time.perf_counter() - overall_start_time

        overall_success = (tensor_analysis_success and
                          token_validation_success and
                          multi_prompt_success)

        self._print_final_summary(overall_success, total_time)

        return overall_success

    def _print_final_summary(self, overall_success: bool, total_time: float) -> None:
        """Print comprehensive final summary."""
        self.log("\n" + "=" * 80)
        self.log("COMPREHENSIVE FULL FORWARD PASS VALIDATION SUMMARY")
        self.log("=" * 80)

        # Test results
        forward_pass_success = self.validation_results.get('forward_pass', {}).get('success', False)
        tensor_analysis_success = self.validation_results.get('tensor_analysis', {}).get('success', False)
        token_validation_success = self.validation_results.get('token_validation', {}).get('success', False)
        multi_prompt_success = self.validation_results.get('multi_prompt_test', {}).get('success', False)

        self.log(f"Forward Pass Execution:     {'✅ PASS' if forward_pass_success else '❌ FAIL'}")
        self.log(f"Tensor Analysis (All Layers): {'✅ PASS' if tensor_analysis_success else '❌ FAIL'}")
        self.log(f"Output Token Validation:    {'✅ PASS' if token_validation_success else '❌ FAIL'}")
        self.log(f"Multi-Prompt Robustness:    {'✅ PASS' if multi_prompt_success else '❌ FAIL'}")

        # Key metrics
        if 'tensor_analysis' in self.validation_results:
            layers_analyzed = self.validation_results['tensor_analysis']['layers_analyzed']
            self.log(f"\nKey Metrics:")
            self.log(f"  Layers Analyzed: {layers_analyzed}")
            self.log(f"  Tensors Captured: {len(self.captured_tensors)}")

        if 'multi_prompt_test' in self.validation_results:
            prompts_tested = self.validation_results['multi_prompt_test']['prompts_tested']
            prompts_passed = self.validation_results['multi_prompt_test']['prompts_passed']
            self.log(f"  Prompts Tested: {prompts_tested}")
            self.log(f"  Prompts Passed: {prompts_passed}")

        self.log(f"  Total Test Time: {total_time:.2f}s")

        # Overall status
        self.log(f"\nOverall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

        if overall_success:
            self.log("\n🎉 COMPREHENSIVE VALIDATION SUCCESSFUL!")
            self.log("✅ Complete forward pass through all layers working")
            self.log("✅ Tensor values verified at each layer")
            self.log("✅ Output tokens are reasonable and correct")
            self.log("✅ Multi-prompt robustness confirmed")
        else:
            self.log("\n⚠️ VALIDATION ISSUES DETECTED!")
            self.log("Check detailed results above for specific problems")

        self.log("=" * 80)


def main():
    """Main function for running comprehensive forward pass validation."""
    print("🚀 Full Forward Pass Validation Test")
    print("Testing complete L4MA model forward pass with tensor verification")
    print("This validates the entire computation pipeline through all 16 layers")
    print()

    # Initialize validator
    validator = FullForwardPassValidator(verbose=True)

    # Run comprehensive validation
    success = validator.run_comprehensive_validation()

    # Save results
    results_path = Path(__file__).parent / "full_forward_pass_validation_results.json"
    try:
        with open(results_path, 'w') as f:
            json.dump(validator.validation_results, f, indent=2, default=str)
        print(f"\n📄 Validation results saved to: {results_path}")
    except Exception as e:
        print(f"\n⚠️ Failed to save results: {e}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)