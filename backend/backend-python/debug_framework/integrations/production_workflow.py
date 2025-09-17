#!/usr/bin/env python3
"""
Production Debug Framework Workflow

Complete end-to-end workflow that integrates L4MA model inference with Metal backend
computational verification, artifact management, and the proven T063-T065 tensor
validation system.

This module provides a production-ready workflow that:
1. Captures tensors during L4MA inference (51-event system)
2. Validates against Metal backend computations
3. Manages artifacts with auto-detection
4. Provides comprehensive reporting
5. Ensures 100% test compatibility with existing validation
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile

# Add backend-python to path
backend_python_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_python_path))

try:
    from debug_framework.integrations.l4ma_real_integration import L4MARealDebugIntegration
    from debug_framework.services.artifact_manager import ArtifactManager
    from debug_framework.services.metal_validator import MetalKernelValidator
    from debug_framework.cli.tensor_detector import TensorDetector
    from debug_framework.models.tensor_recording import TensorRecording
    WORKFLOW_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Workflow components not available: {e}")
    WORKFLOW_COMPONENTS_AVAILABLE = False

# Import test integration for tensor capture
try:
    import sys
    integration_test_path = Path(__file__).parent.parent.parent.parent / "tests" / "debug_framework" / "integration"
    sys.path.insert(0, str(integration_test_path))
    from test_l4ma_backend_integration import BackendReuseIntegrationTest
    BACKEND_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Backend integration test not available: {e}")
    BACKEND_INTEGRATION_AVAILABLE = False


@dataclass
class WorkflowConfig:
    """Configuration for the production workflow."""
    session_name: str
    model_name: str = "llama-3.2-1b-instruct"
    enable_metal_validation: bool = True
    enable_artifact_management: bool = True
    enable_auto_detection: bool = True
    tensor_tolerance: float = 1e-5
    max_capture_events: int = 51
    storage_dir: Optional[str] = None
    performance_profiling: bool = True


@dataclass
class WorkflowResult:
    """Result of the complete workflow execution."""
    session_id: int
    success: bool
    l4ma_inference_success: bool
    tensor_capture_count: int
    metal_validation_success: bool
    artifact_count: int
    auto_detected_count: int
    validation_pass_rate: float
    total_execution_time: float
    error_messages: List[str]
    summary_report: Dict[str, Any]


class ProductionDebugWorkflow:
    """
    Production-ready debug framework workflow.

    Integrates all components of the debug framework to provide a complete
    end-to-end validation workflow from L4MA inference to Metal verification.
    """

    def __init__(self, config: WorkflowConfig):
        """Initialize the production workflow with configuration."""
        self.config = config
        self.workflow_id = f"workflow_{int(time.time() * 1000000)}"

        # Initialize components
        self.artifact_manager = None
        self.metal_validator = None
        self.tensor_detector = None
        self.l4ma_integration = None
        self.backend_test = None

        # Workflow state
        self.session_id: Optional[int] = None
        self.captured_tensors: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        self.error_messages: List[str] = []

        # Performance tracking
        self.start_time = 0.0
        self.execution_times = {}

        logging.info(f"ProductionDebugWorkflow initialized: {self.workflow_id}")

    def execute_complete_workflow(self) -> WorkflowResult:
        """
        Execute the complete production debug workflow.

        Returns:
            Comprehensive workflow result
        """
        print("🚀 Starting Production Debug Framework Workflow")
        print("=" * 70)
        print(f"Workflow ID: {self.workflow_id}")
        print(f"Session: {self.config.session_name}")
        print(f"Model: {self.config.model_name}")
        print("=" * 70)

        self.start_time = time.perf_counter()

        try:
            # Step 1: Initialize all components
            step_success = self._initialize_components()
            if not step_success:
                return self._create_failure_result("Component initialization failed")

            # Step 2: Run L4MA inference with tensor capture
            step_success = self._execute_l4ma_inference()
            if not step_success:
                return self._create_failure_result("L4MA inference failed")

            # Step 3: Metal backend validation (if enabled)
            metal_validation_success = True
            if self.config.enable_metal_validation:
                metal_validation_success = self._execute_metal_validation()

            # Step 4: Auto-detect additional tensor recordings (if enabled)
            auto_detected_count = 0
            if self.config.enable_auto_detection:
                auto_detected_count = self._execute_auto_detection()

            # Step 5: Generate comprehensive report
            summary_report = self._generate_workflow_report()

            # Step 6: Finalize session
            self._finalize_workflow_session()

            total_time = time.perf_counter() - self.start_time

            # Create success result
            return WorkflowResult(
                session_id=self.session_id,
                success=True,
                l4ma_inference_success=True,
                tensor_capture_count=len(self.captured_tensors),
                metal_validation_success=metal_validation_success,
                artifact_count=self._get_artifact_count(),
                auto_detected_count=auto_detected_count,
                validation_pass_rate=self._calculate_validation_pass_rate(),
                total_execution_time=total_time,
                error_messages=self.error_messages.copy(),
                summary_report=summary_report
            )

        except Exception as e:
            self.error_messages.append(str(e))
            logging.error(f"Workflow execution failed: {e}")
            return self._create_failure_result(f"Workflow execution failed: {e}")

    def _initialize_components(self) -> bool:
        """Initialize all workflow components."""
        print("\n📋 Step 1: Initializing Components")
        print("-" * 40)

        try:
            start_time = time.perf_counter()

            # Initialize artifact manager
            if self.config.enable_artifact_management:
                print("  🗂️  Initializing artifact manager...")
                self.artifact_manager = ArtifactManager(
                    base_storage_dir=self.config.storage_dir,
                    enable_auto_detection=self.config.enable_auto_detection
                )

                # Create workflow session
                self.session_id = self.artifact_manager.create_session(
                    session_name=self.config.session_name,
                    model_name=self.config.model_name,
                    metadata={
                        'workflow_id': self.workflow_id,
                        'workflow_type': 'production_debug_framework',
                        'enable_metal_validation': self.config.enable_metal_validation,
                        'tensor_tolerance': self.config.tensor_tolerance,
                        'max_capture_events': self.config.max_capture_events
                    }
                )
                print(f"     ✅ Session created: {self.session_id}")

            # Initialize Metal validator
            if self.config.enable_metal_validation:
                print("  🔧 Initializing Metal validator...")
                self.metal_validator = MetalKernelValidator(
                    artifact_manager=self.artifact_manager,
                    tolerance=self.config.tensor_tolerance,
                    enable_performance_profiling=self.config.performance_profiling
                )
                print(f"     ✅ Metal validator ready (available: {self.metal_validator.metal_available})")

            # Initialize tensor detector
            if self.config.enable_auto_detection:
                print("  🔍 Initializing tensor detector...")
                self.tensor_detector = TensorDetector(
                    storage_dir=self.config.storage_dir,
                    verbose=False
                )
                print("     ✅ Tensor detector ready")

            # Initialize L4MA integration
            print("  🧠 Initializing L4MA debug integration...")
            if not BACKEND_INTEGRATION_AVAILABLE:
                raise RuntimeError("Backend integration not available")

            self.backend_test = BackendReuseIntegrationTest()

            # Load model
            if not self.backend_test.load_model_with_backend_handler():
                raise RuntimeError("Failed to load L4MA model")

            # Set up debug integration
            if not self.backend_test.integrate_with_debug_framework():
                raise RuntimeError("Failed to setup debug framework integration")

            # Set up tensor capture callback
            self.backend_test.debug_integration.set_tensor_capture_callback(
                self._tensor_capture_callback
            )

            print("     ✅ L4MA integration ready")

            self.execution_times['initialization'] = time.perf_counter() - start_time
            print(f"  ⏱️  Initialization completed in {self.execution_times['initialization']:.2f}s")
            return True

        except Exception as e:
            self.error_messages.append(f"Component initialization failed: {e}")
            print(f"  ❌ Initialization failed: {e}")
            return False

    def _execute_l4ma_inference(self) -> bool:
        """Execute L4MA inference with tensor capture."""
        print("\n🧠 Step 2: L4MA Inference with Tensor Capture")
        print("-" * 40)

        try:
            start_time = time.perf_counter()

            # Clear previous captures
            self.captured_tensors.clear()

            # Run inference with a meaningful prompt
            test_prompt = "The capital of France is"
            print(f"  📝 Running inference with prompt: '{test_prompt}'")

            # Enable tensor capture
            self.backend_test.debug_integration.enable_real_tensor_validation(True)

            # Execute inference
            result = self.backend_test.test_prompt_inference_with_handler(test_prompt)

            if not result.get('success', False):
                raise RuntimeError(f"Inference failed: {result.get('error', 'unknown')}")

            # Validate captured tensors
            capture_count = len(self.captured_tensors)
            print(f"  ✅ Inference completed successfully")
            print(f"  📊 Captured {capture_count} tensor events")

            if capture_count > 0:
                print(f"     Sample tensor shapes: {[c['tensor_shape'] for c in self.captured_tensors[:3]]}")
                print(f"     Event types: {set(c['checkpoint_name'] for c in self.captured_tensors)}")
            else:
                print("  ⚠️  Warning: No tensors captured during inference")

            # Store captured tensors as artifacts
            if self.artifact_manager and self.session_id:
                stored_count = self._store_captured_tensors()
                print(f"  🗂️  Stored {stored_count} tensor recordings as artifacts")

            self.execution_times['l4ma_inference'] = time.perf_counter() - start_time
            print(f"  ⏱️  L4MA inference completed in {self.execution_times['l4ma_inference']:.2f}s")
            return True

        except Exception as e:
            self.error_messages.append(f"L4MA inference failed: {e}")
            print(f"  ❌ L4MA inference failed: {e}")
            return False

    def _execute_metal_validation(self) -> bool:
        """Execute Metal backend validation."""
        print("\n🔧 Step 3: Metal Backend Validation")
        print("-" * 40)

        try:
            start_time = time.perf_counter()

            if not self.metal_validator:
                print("  ⏭️  Metal validation disabled")
                return True

            print("  🧪 Running comprehensive Metal kernel validation...")

            # Run Metal validation
            validation_report = self.metal_validator.run_comprehensive_validation(
                session_name=f"Metal Validation - {self.config.session_name}",
                model_name=self.config.model_name
            )

            # Store validation results
            self.validation_results = validation_report.get('detailed_results', [])

            # Check validation success
            overall_pass_rate = validation_report.get('overall_pass_rate', 0.0)
            validation_success = overall_pass_rate >= 0.8  # 80% threshold

            print(f"  📊 Validation Results:")
            print(f"     Overall pass rate: {overall_pass_rate*100:.1f}%")
            print(f"     Total tests: {validation_report.get('total_tests', 0)}")
            print(f"     Passed tests: {validation_report.get('passed_tests', 0)}")

            status = "✅ PASS" if validation_success else "❌ FAIL"
            print(f"  {status} Metal validation completed")

            self.execution_times['metal_validation'] = time.perf_counter() - start_time
            print(f"  ⏱️  Metal validation completed in {self.execution_times['metal_validation']:.2f}s")

            return validation_success

        except Exception as e:
            self.error_messages.append(f"Metal validation failed: {e}")
            print(f"  ❌ Metal validation failed: {e}")
            return False

    def _execute_auto_detection(self) -> int:
        """Execute auto-detection of additional tensor recordings."""
        print("\n🔍 Step 4: Auto-Detection of Tensor Recordings")
        print("-" * 40)

        try:
            start_time = time.perf_counter()

            if not self.tensor_detector or not self.artifact_manager:
                print("  ⏭️  Auto-detection disabled")
                return 0

            # Detect tensor recordings
            print("  🔍 Scanning for tensor recordings...")
            detected_files = self.tensor_detector.auto_detect_tensor_recordings()

            if not detected_files:
                print("  📭 No additional tensor recordings detected")
                return 0

            print(f"  📁 Detected {len(detected_files)} potential tensor files")

            # Organize detected recordings
            organization_result = self.tensor_detector.organize_recordings(
                detected_files,
                session_name=f"Auto-detected - {self.config.session_name}",
                model_name=self.config.model_name,
                min_confidence=0.5
            )

            organized_count = organization_result.get('organized', 0)
            print(f"  🗂️  Organized {organized_count} recordings into artifacts")

            self.execution_times['auto_detection'] = time.perf_counter() - start_time
            print(f"  ⏱️  Auto-detection completed in {self.execution_times['auto_detection']:.2f}s")

            return organized_count

        except Exception as e:
            self.error_messages.append(f"Auto-detection failed: {e}")
            print(f"  ❌ Auto-detection failed: {e}")
            return 0

    def _generate_workflow_report(self) -> Dict[str, Any]:
        """Generate comprehensive workflow report."""
        print("\n📊 Step 5: Generating Workflow Report")
        print("-" * 40)

        try:
            # Collect storage stats
            storage_stats = {}
            if self.artifact_manager:
                storage_stats = self.artifact_manager.get_storage_stats()

            # Calculate metrics
            total_time = time.perf_counter() - self.start_time
            validation_pass_rate = self._calculate_validation_pass_rate()

            # Create comprehensive report
            report = {
                'workflow_metadata': {
                    'workflow_id': self.workflow_id,
                    'session_id': self.session_id,
                    'session_name': self.config.session_name,
                    'model_name': self.config.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'total_execution_time': total_time
                },
                'execution_summary': {
                    'l4ma_inference_success': len(self.captured_tensors) > 0,
                    'tensor_capture_count': len(self.captured_tensors),
                    'metal_validation_success': validation_pass_rate >= 0.8,
                    'validation_pass_rate': validation_pass_rate,
                    'artifact_count': self._get_artifact_count(),
                    'error_count': len(self.error_messages)
                },
                'performance_metrics': {
                    'total_time': total_time,
                    'execution_times': self.execution_times.copy(),
                    'throughput_tensors_per_second': len(self.captured_tensors) / total_time if total_time > 0 else 0
                },
                'storage_metrics': storage_stats,
                'validation_details': {
                    'tensor_tolerance': self.config.tensor_tolerance,
                    'max_capture_events': self.config.max_capture_events,
                    'validation_results': self.validation_results
                },
                'error_messages': self.error_messages.copy()
            }

            print(f"  📋 Report generated successfully")
            print(f"     Total execution time: {total_time:.2f}s")
            print(f"     Tensor capture count: {len(self.captured_tensors)}")
            print(f"     Validation pass rate: {validation_pass_rate*100:.1f}%")

            return report

        except Exception as e:
            self.error_messages.append(f"Report generation failed: {e}")
            print(f"  ❌ Report generation failed: {e}")
            return {}

    def _finalize_workflow_session(self) -> None:
        """Finalize the workflow session."""
        print("\n🏁 Step 6: Finalizing Workflow Session")
        print("-" * 40)

        try:
            if self.artifact_manager and self.session_id:
                # Mark session as completed
                self.artifact_manager.finish_session(self.session_id)
                print(f"  ✅ Session {self.session_id} marked as completed")

            # Clean up components
            if self.backend_test and hasattr(self.backend_test, 'debug_integration'):
                self.backend_test.debug_integration.cleanup_and_restore()
                print("  🧹 L4MA integration cleaned up")

            print("  ✅ Workflow session finalized")

        except Exception as e:
            self.error_messages.append(f"Session finalization failed: {e}")
            print(f"  ⚠️  Session finalization warning: {e}")

    def _tensor_capture_callback(self, checkpoint_name: str, tensor_data: Any, metadata: Dict[str, Any]):
        """Callback for capturing tensor data during inference."""
        try:
            # Extract tensor information
            capture_info = {
                'checkpoint_name': checkpoint_name,
                'tensor_shape': getattr(tensor_data, 'shape', (0,)),
                'tensor_dtype': str(getattr(tensor_data, 'dtype', 'unknown')),
                'metadata': metadata.copy(),
                'timestamp': time.perf_counter()
            }

            # Store the actual tensor data and compute memory hash for validation
            if hasattr(tensor_data, 'numpy'):
                # PyTorch tensor
                numpy_data = tensor_data.detach().cpu().numpy()
                capture_info['tensor_data'] = numpy_data.copy()  # Store actual data
                capture_info['memory_hash'] = self._compute_tensor_hash(numpy_data)
                capture_info['tensor_size_bytes'] = numpy_data.nbytes
            elif hasattr(tensor_data, 'tobytes'):
                # NumPy array
                capture_info['tensor_data'] = tensor_data.copy()  # Store actual data
                capture_info['memory_hash'] = self._compute_tensor_hash(tensor_data)
                capture_info['tensor_size_bytes'] = tensor_data.nbytes

            self.captured_tensors.append(capture_info)

            # Limit capture count
            if len(self.captured_tensors) >= self.config.max_capture_events:
                print(f"  ⚠️  Reached maximum capture events ({self.config.max_capture_events})")

        except Exception as e:
            self.error_messages.append(f"Tensor capture failed: {e}")

    def _store_captured_tensors(self) -> int:
        """Store captured tensors as artifacts."""
        if not self.artifact_manager or not self.session_id:
            return 0

        stored_count = 0
        for i, capture_info in enumerate(self.captured_tensors[:10]):  # Store first 10 for demo
            try:
                # Use the actual captured tensor data instead of synthetic data
                if 'tensor_data' not in capture_info:
                    print(f"  ⚠️  No tensor data available for {capture_info['checkpoint_name']}, skipping")
                    continue

                actual_tensor = capture_info['tensor_data']

                tensor_recording = TensorRecording.create_from_tensor(
                    session_id=self.session_id,
                    checkpoint_id=i + 1,
                    tensor_name=f"{capture_info['checkpoint_name']}_tensor_{i}",
                    tensor_data=actual_tensor,
                    backend_name='l4ma_workflow',
                    device_info={
                        'platform': 'cpu',
                        'device': 'cpu'
                    },
                    storage_dir=str(self.artifact_manager.temp_dir)
                )

                artifact_id = self.artifact_manager.store_tensor_recording(
                    tensor_recording,
                    self.session_id,
                    tags=['workflow_capture', capture_info['checkpoint_name']]
                )

                stored_count += 1

            except Exception as e:
                self.error_messages.append(f"Failed to store tensor {i}: {e}")

        return stored_count

    def _compute_tensor_hash(self, tensor_data) -> str:
        """Compute SHA256 hash of tensor data."""
        try:
            import hashlib
            if hasattr(tensor_data, 'tobytes'):
                return hashlib.sha256(tensor_data.tobytes()).hexdigest()
            else:
                return ""
        except Exception:
            return ""

    def _get_artifact_count(self) -> int:
        """Get total artifact count for the session."""
        if not self.artifact_manager or not self.session_id:
            return 0

        try:
            artifacts = self.artifact_manager.get_session_artifacts(self.session_id)
            return len(artifacts)
        except Exception:
            return 0

    def _calculate_validation_pass_rate(self) -> float:
        """Calculate overall validation pass rate."""
        if not self.validation_results:
            return 1.0  # No validation failures

        passed = len([r for r in self.validation_results if r.get('success', False)])
        total = len(self.validation_results)

        return passed / total if total > 0 else 1.0

    def _create_failure_result(self, error_message: str) -> WorkflowResult:
        """Create a failure result with error information."""
        total_time = time.perf_counter() - self.start_time

        return WorkflowResult(
            session_id=self.session_id or 0,
            success=False,
            l4ma_inference_success=False,
            tensor_capture_count=len(self.captured_tensors),
            metal_validation_success=False,
            artifact_count=self._get_artifact_count(),
            auto_detected_count=0,
            validation_pass_rate=0.0,
            total_execution_time=total_time,
            error_messages=self.error_messages + [error_message],
            summary_report={}
        )

    def print_workflow_summary(self, result: WorkflowResult) -> None:
        """Print comprehensive workflow summary."""
        print("\n" + "=" * 70)
        print("PRODUCTION DEBUG FRAMEWORK WORKFLOW SUMMARY")
        print("=" * 70)
        print(f"Workflow ID: {self.workflow_id}")
        print(f"Session ID: {result.session_id}")
        print(f"Execution Time: {result.total_execution_time:.2f}s")
        print()

        # Overall status
        overall_status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"Overall Status: {overall_status}")
        print()

        # Component status
        print("Component Results:")
        l4ma_status = "✅" if result.l4ma_inference_success else "❌"
        metal_status = "✅" if result.metal_validation_success else "❌"

        print(f"  {l4ma_status} L4MA Inference: {result.tensor_capture_count} tensors captured")
        print(f"  {metal_status} Metal Validation: {result.validation_pass_rate*100:.1f}% pass rate")
        print(f"  📁 Artifact Management: {result.artifact_count} artifacts stored")
        print(f"  🔍 Auto-Detection: {result.auto_detected_count} recordings detected")
        print()

        # Error summary
        if result.error_messages:
            print("Errors:")
            for i, error in enumerate(result.error_messages[:5], 1):
                print(f"  {i}. {error}")
            if len(result.error_messages) > 5:
                print(f"  ... and {len(result.error_messages) - 5} more errors")
        else:
            print("✅ No errors reported")

        print("=" * 70)


def create_production_workflow(
    session_name: str,
    model_name: str = "llama-3.2-1b-instruct",
    **kwargs
) -> ProductionDebugWorkflow:
    """
    Factory function to create a production debug workflow.

    Args:
        session_name: Name for the debug session
        model_name: Model name for validation
        **kwargs: Additional configuration options

    Returns:
        Configured ProductionDebugWorkflow instance
    """
    config = WorkflowConfig(
        session_name=session_name,
        model_name=model_name,
        **kwargs
    )

    return ProductionDebugWorkflow(config)


def main():
    """Demo/test function for the production workflow."""
    # Create workflow configuration
    config = WorkflowConfig(
        session_name="Production Debug Framework Demo",
        model_name="llama-3.2-1b-instruct",
        enable_metal_validation=True,
        enable_artifact_management=True,
        enable_auto_detection=True,
        tensor_tolerance=1e-5
    )

    # Create and execute workflow
    workflow = ProductionDebugWorkflow(config)
    result = workflow.execute_complete_workflow()

    # Print summary
    workflow.print_workflow_summary(result)

    return result.success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)