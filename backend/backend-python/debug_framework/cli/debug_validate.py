#!/usr/bin/env python3
"""
Debug Validate CLI Tool

Validates kernel implementations across different backends using
the debug framework's validation engine.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from debug_framework.services.validation_engine import ValidationEngine
from debug_framework.services.database_manager import DatabaseManager
from debug_framework.integrations.metal_backend import MetalBackend
from debug_framework.integrations.l4ma_python_backend import L4MAPythonBackend


class DebugValidateCLI:
    """CLI tool for validating kernel implementations."""

    def __init__(self):
        self.validation_engine: Optional[ValidationEngine] = None
        self.results: List[Dict[str, Any]] = []

    def initialize_validation_engine(self, database_path: Optional[str] = None) -> bool:
        """Initialize validation engine with database."""
        try:
            db_path = database_path or "/tmp/debug_validate.db"
            db_manager = DatabaseManager(db_path)
            self.validation_engine = ValidationEngine(database_manager=db_manager)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize validation engine: {e}", file=sys.stderr)
            return False

    def validate_metal_kernels(self, metal_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate Metal kernel implementations."""
        print("🔍 Validating Metal kernels...")

        try:
            metal_backend = MetalBackend(metal_path)

            if not metal_backend.initialize():
                return {
                    "backend": "metal",
                    "status": "failed",
                    "error": "Metal backend initialization failed",
                    "capabilities": metal_backend.get_capabilities()
                }

            # Test available operations
            operations_tested = []
            operations_passed = []

            capabilities = metal_backend.get_capabilities()
            available_kernels = capabilities.get('available_kernels', {})

            # Test softmax if available
            if available_kernels.get('softmax', False):
                operations_tested.append('softmax')
                try:
                    import numpy as np
                    test_input = np.random.randn(4, 1024).astype(np.float32)
                    result = metal_backend.execute_softmax(test_input)
                    if result is not None and result.shape == test_input.shape:
                        operations_passed.append('softmax')
                        print("  ✅ Softmax kernel validation passed")
                    else:
                        print("  ❌ Softmax kernel validation failed: invalid output shape")
                except Exception as e:
                    print(f"  ❌ Softmax kernel validation failed: {e}")

            # Test attention if available
            if available_kernels.get('attention', False):
                operations_tested.append('attention')
                try:
                    import numpy as np
                    batch_size, seq_len, head_size = 1, 16, 64
                    query = np.random.randn(batch_size, seq_len, head_size).astype(np.float32)
                    key = np.random.randn(batch_size, seq_len, head_size).astype(np.float32)
                    value = np.random.randn(batch_size, seq_len, head_size).astype(np.float32)

                    result = metal_backend.run_attention(
                        query, key, value,
                        num_query_heads=1,
                        num_kv_heads=1,
                        head_size=head_size,
                        page_size=16
                    )

                    if result and result.output is not None:
                        operations_passed.append('attention')
                        print("  ✅ Attention kernel validation passed")
                    else:
                        print("  ❌ Attention kernel validation failed: no output")
                except Exception as e:
                    print(f"  ❌ Attention kernel validation failed: {e}")

            metal_backend.cleanup()

            success_rate = len(operations_passed) / max(len(operations_tested), 1) * 100

            return {
                "backend": "metal",
                "status": "passed" if success_rate == 100.0 else "partial",
                "success_rate": success_rate,
                "operations_tested": operations_tested,
                "operations_passed": operations_passed,
                "capabilities": capabilities,
                "device_info": capabilities.get('device_info', 'Unknown')
            }

        except Exception as e:
            return {
                "backend": "metal",
                "status": "failed",
                "error": str(e)
            }

    def validate_reference_backend(self) -> Dict[str, Any]:
        """Validate reference Python backend."""
        print("🔍 Validating reference Python backend...")

        try:
            ref_backend = L4MAPythonBackend()

            if not ref_backend.initialize():
                return {
                    "backend": "python_reference",
                    "status": "failed",
                    "error": "Reference backend initialization failed"
                }

            operations_tested = []
            operations_passed = []

            # Test basic tensor operations
            try:
                import numpy as np

                # Test attention
                operations_tested.append('attention')
                batch_size, seq_len, head_size = 1, 8, 32
                query = np.random.randn(batch_size, seq_len, head_size).astype(np.float32)
                key = np.random.randn(batch_size, seq_len, head_size).astype(np.float32)
                value = np.random.randn(batch_size, seq_len, head_size).astype(np.float32)

                result = ref_backend.run_attention(query, key, value)
                if result and result.output is not None:
                    operations_passed.append('attention')
                    print("  ✅ Reference attention validation passed")

            except Exception as e:
                print(f"  ❌ Reference attention validation failed: {e}")

            success_rate = len(operations_passed) / max(len(operations_tested), 1) * 100

            return {
                "backend": "python_reference",
                "status": "passed" if success_rate == 100.0 else "partial",
                "success_rate": success_rate,
                "operations_tested": operations_tested,
                "operations_passed": operations_passed
            }

        except Exception as e:
            return {
                "backend": "python_reference",
                "status": "failed",
                "error": str(e)
            }

    def run_comparative_validation(self, backends: List[str]) -> Dict[str, Any]:
        """Run comparative validation between backends."""
        print("🔍 Running comparative validation...")

        if not self.validation_engine:
            return {"status": "failed", "error": "Validation engine not initialized"}

        try:
            # Create a validation session
            session_id = self.validation_engine.create_session(
                model_path="/tmp/test_model",
                config={
                    "enabled_checkpoints": ["post_attention", "post_processing"],
                    "precision_thresholds": {"rtol": 1e-4, "atol": 1e-6}
                },
                reference_backend=backends[0] if backends else "python_reference",
                alternative_backend=backends[1] if len(backends) > 1 else "metal"
            )

            # Run validation workflow
            results = asyncio.run(self.validation_engine.execute_validation_workflow(session_id))

            self.validation_engine.complete_session(session_id)
            self.validation_engine.cleanup_session(session_id)

            return {
                "status": "completed",
                "session_id": session_id,
                "results": results,
                "backends": backends
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def generate_report(self, output_format: str = "json", output_file: Optional[str] = None) -> None:
        """Generate validation report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_results": self.results,
            "summary": {
                "total_backends_tested": len(self.results),
                "passed": len([r for r in self.results if r.get("status") == "passed"]),
                "failed": len([r for r in self.results if r.get("status") == "failed"]),
                "partial": len([r for r in self.results if r.get("status") == "partial"])
            }
        }

        if output_format == "json":
            report_content = json.dumps(report, indent=2)
        else:
            # Text format
            lines = [
                f"Debug Framework Validation Report",
                f"Generated: {report['timestamp']}",
                f"",
                f"Summary:",
                f"  Total backends tested: {report['summary']['total_backends_tested']}",
                f"  Passed: {report['summary']['passed']}",
                f"  Failed: {report['summary']['failed']}",
                f"  Partial: {report['summary']['partial']}",
                f"",
                f"Results:"
            ]

            for result in self.results:
                lines.append(f"  Backend: {result.get('backend', 'unknown')}")
                lines.append(f"    Status: {result.get('status', 'unknown')}")
                if 'success_rate' in result:
                    lines.append(f"    Success Rate: {result['success_rate']:.1f}%")
                if 'operations_passed' in result:
                    lines.append(f"    Operations Passed: {', '.join(result['operations_passed'])}")
                if 'error' in result:
                    lines.append(f"    Error: {result['error']}")
                lines.append("")

            report_content = '\n'.join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"📄 Report saved to {output_file}")
        else:
            print(report_content)

    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="Debug Framework Kernel Validation Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Validate Metal kernels
  debug-validate --backend metal

  # Validate all available backends
  debug-validate --backend all

  # Run comparative validation
  debug-validate --backend metal,python_reference --compare

  # Generate JSON report
  debug-validate --backend all --output report.json --format json
            """
        )

        parser.add_argument(
            "--backend", "-b",
            default="all",
            help="Backend to validate (metal, python_reference, all, or comma-separated list)"
        )

        parser.add_argument(
            "--compare", "-c",
            action="store_true",
            help="Run comparative validation between backends"
        )

        parser.add_argument(
            "--metal-path",
            help="Path to Metal backend directory"
        )

        parser.add_argument(
            "--database", "-d",
            help="Path to debug database file"
        )

        parser.add_argument(
            "--output", "-o",
            help="Output file for validation report"
        )

        parser.add_argument(
            "--format", "-f",
            choices=["json", "text"],
            default="text",
            help="Report output format"
        )

        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )

        args = parser.parse_args()

        # Initialize validation engine
        if not self.initialize_validation_engine(args.database):
            sys.exit(1)

        # Parse backend selection
        if args.backend == "all":
            backends = ["metal", "python_reference"]
        else:
            backends = [b.strip() for b in args.backend.split(",")]

        print(f"🚀 Starting debug framework validation")
        print(f"   Backends: {', '.join(backends)}")
        print()

        # Run validations
        for backend in backends:
            if backend == "metal":
                result = self.validate_metal_kernels(args.metal_path)
            elif backend == "python_reference":
                result = self.validate_reference_backend()
            else:
                result = {
                    "backend": backend,
                    "status": "failed",
                    "error": f"Unknown backend: {backend}"
                }

            self.results.append(result)

            # Print immediate feedback
            status_emoji = {
                "passed": "✅",
                "partial": "⚠️",
                "failed": "❌"
            }.get(result.get("status"), "❓")

            print(f"{status_emoji} {result.get('backend', 'unknown')}: {result.get('status', 'unknown')}")

            if args.verbose and "error" in result:
                print(f"   Error: {result['error']}")
            if args.verbose and "success_rate" in result:
                print(f"   Success Rate: {result['success_rate']:.1f}%")

        # Run comparative validation if requested
        if args.compare and len(backends) > 1:
            print()
            comparison_result = self.run_comparative_validation(backends)
            self.results.append(comparison_result)

            status_emoji = "✅" if comparison_result.get("status") == "completed" else "❌"
            print(f"{status_emoji} Comparative validation: {comparison_result.get('status')}")

        print()

        # Generate report
        self.generate_report(args.format, args.output)

        # Exit with appropriate code
        failed_count = len([r for r in self.results if r.get("status") == "failed"])
        sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    cli = DebugValidateCLI()
    cli.main()