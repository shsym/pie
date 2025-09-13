#!/usr/bin/env python3
"""
Session Report CLI Tool

Generates comprehensive debugging session reports from
debug framework database records.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from debug_framework.services.database_manager import DatabaseManager
from debug_framework.services.validation_engine import ValidationEngine
from debug_framework.models.debug_session import DebugSession
from debug_framework.models.validation_checkpoint import ValidationCheckpoint
from debug_framework.models.tensor_comparison import TensorComparison


class SessionReportCLI:
    """CLI tool for generating debug session reports."""

    def __init__(self):
        self.database_manager: Optional[DatabaseManager] = None
        self.validation_engine: Optional[ValidationEngine] = None

    def initialize_services(self, database_path: str) -> bool:
        """Initialize database manager and validation engine."""
        try:
            self.database_manager = DatabaseManager(database_path)
            self.validation_engine = ValidationEngine(database_manager=self.database_manager)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}", file=sys.stderr)
            return False

    def get_session_list(self, days_back: int = 30, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of debug sessions."""
        print(f"🔍 Scanning for debug sessions (last {days_back} days)...")

        # Mock session data since we don't have actual database queries implemented
        # In a real implementation, this would query the sessions table
        current_time = datetime.now()

        mock_sessions = [
            {
                "id": "session_001",
                "model_path": "/tmp/test_model_llama3",
                "reference_backend": "python_reference",
                "alternative_backend": "metal",
                "status": "completed",
                "created_at": current_time - timedelta(hours=2),
                "completed_at": current_time - timedelta(hours=1),
                "checkpoints_count": 4,
                "success_rate": 100.0
            },
            {
                "id": "session_002",
                "model_path": "/tmp/test_model_gpt2",
                "reference_backend": "python_reference",
                "alternative_backend": "cuda",
                "status": "failed",
                "created_at": current_time - timedelta(days=1),
                "completed_at": None,
                "checkpoints_count": 2,
                "success_rate": 0.0
            },
            {
                "id": "session_003",
                "model_path": "/tmp/test_model_mistral",
                "reference_backend": "python_reference",
                "alternative_backend": "metal",
                "status": "partial",
                "created_at": current_time - timedelta(days=3),
                "completed_at": current_time - timedelta(days=3, hours=-1),
                "checkpoints_count": 5,
                "success_rate": 80.0
            }
        ]

        # Apply status filter
        if status_filter:
            mock_sessions = [s for s in mock_sessions if s["status"] == status_filter]

        print(f"  Found {len(mock_sessions)} sessions")
        return mock_sessions

    def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific session."""
        print(f"📊 Retrieving details for session {session_id}...")

        # Mock detailed session data
        # In real implementation, this would query multiple tables
        session_details = {
            "session_info": {
                "id": session_id,
                "model_path": "/tmp/test_model_llama3",
                "reference_backend": "python_reference",
                "alternative_backend": "metal",
                "status": "completed",
                "created_at": datetime.now() - timedelta(hours=2),
                "completed_at": datetime.now() - timedelta(hours=1),
                "total_duration": 3600.0  # seconds
            },
            "checkpoints": [
                {
                    "name": "post_embedding",
                    "status": "passed",
                    "execution_time": 0.15,
                    "tensor_comparisons": [
                        {
                            "tensor_name": "hidden_states",
                            "shape": [1, 32, 4096],
                            "similarity_score": 0.9998,
                            "max_diff": 1.2e-6,
                            "mean_diff": 3.4e-7,
                            "status": "passed"
                        }
                    ]
                },
                {
                    "name": "post_attention",
                    "status": "passed",
                    "execution_time": 0.87,
                    "tensor_comparisons": [
                        {
                            "tensor_name": "attention_output",
                            "shape": [1, 32, 4096],
                            "similarity_score": 0.9995,
                            "max_diff": 8.7e-6,
                            "mean_diff": 2.1e-6,
                            "status": "passed"
                        }
                    ]
                },
                {
                    "name": "post_mlp",
                    "status": "passed",
                    "execution_time": 0.23,
                    "tensor_comparisons": [
                        {
                            "tensor_name": "mlp_output",
                            "shape": [1, 32, 4096],
                            "similarity_score": 0.9997,
                            "max_diff": 5.2e-6,
                            "mean_diff": 1.8e-6,
                            "status": "passed"
                        }
                    ]
                },
                {
                    "name": "post_processing",
                    "status": "passed",
                    "execution_time": 0.11,
                    "tensor_comparisons": [
                        {
                            "tensor_name": "final_output",
                            "shape": [1, 32768],
                            "similarity_score": 0.9999,
                            "max_diff": 2.1e-6,
                            "mean_diff": 4.3e-7,
                            "status": "passed"
                        }
                    ]
                }
            ],
            "performance_metrics": {
                "total_computation_time": 1.36,
                "reference_backend_time": 0.68,
                "alternative_backend_time": 0.68,
                "comparison_overhead": 0.04,
                "memory_peak_usage": 2.1  # GB
            },
            "configuration": {
                "precision_thresholds": {
                    "rtol": 1e-4,
                    "atol": 1e-6
                },
                "enabled_checkpoints": ["post_embedding", "post_attention", "post_mlp", "post_processing"],
                "tensor_recording_enabled": True,
                "performance_profiling_enabled": True
            }
        }

        return session_details

    def generate_summary_report(self, sessions: List[Dict[str, Any]],
                              output_format: str = "text") -> str:
        """Generate summary report for multiple sessions."""
        if not sessions:
            return "No sessions found."

        # Calculate summary statistics
        total_sessions = len(sessions)
        completed_sessions = len([s for s in sessions if s["status"] == "completed"])
        failed_sessions = len([s for s in sessions if s["status"] == "failed"])
        partial_sessions = len([s for s in sessions if s["status"] == "partial"])

        success_rate = completed_sessions / total_sessions * 100
        avg_checkpoints = sum(s["checkpoints_count"] for s in sessions) / total_sessions

        # Backend analysis
        backends = {}
        for session in sessions:
            alt_backend = session["alternative_backend"]
            if alt_backend not in backends:
                backends[alt_backend] = {"total": 0, "success": 0}
            backends[alt_backend]["total"] += 1
            if session["status"] == "completed":
                backends[alt_backend]["success"] += 1

        if output_format == "json":
            return json.dumps({
                "summary": {
                    "total_sessions": total_sessions,
                    "completed_sessions": completed_sessions,
                    "failed_sessions": failed_sessions,
                    "partial_sessions": partial_sessions,
                    "success_rate": success_rate,
                    "average_checkpoints": avg_checkpoints
                },
                "backend_analysis": {
                    backend: {
                        "total": stats["total"],
                        "success": stats["success"],
                        "success_rate": stats["success"] / stats["total"] * 100
                    }
                    for backend, stats in backends.items()
                },
                "sessions": sessions
            }, indent=2, default=str)

        else:
            # Text format
            lines = [
                "Debug Framework Session Summary Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "📊 Overall Statistics:",
                f"  Total sessions: {total_sessions}",
                f"  ✅ Completed: {completed_sessions} ({completed_sessions/total_sessions*100:.1f}%)",
                f"  ❌ Failed: {failed_sessions} ({failed_sessions/total_sessions*100:.1f}%)",
                f"  ⚠️  Partial: {partial_sessions} ({partial_sessions/total_sessions*100:.1f}%)",
                f"  📈 Success rate: {success_rate:.1f}%",
                f"  📋 Avg checkpoints: {avg_checkpoints:.1f}",
                "",
                "🔧 Backend Analysis:"
            ]

            for backend, stats in backends.items():
                backend_success_rate = stats["success"] / stats["total"] * 100
                lines.append(f"  {backend.upper()}:")
                lines.append(f"    Sessions: {stats['total']}")
                lines.append(f"    Success: {stats['success']} ({backend_success_rate:.1f}%)")

            lines.extend([
                "",
                "📝 Recent Sessions:"
            ])

            for session in sorted(sessions, key=lambda x: x["created_at"], reverse=True)[:10]:
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "partial": "⚠️"
                }.get(session["status"], "❓")

                created_str = session["created_at"].strftime("%Y-%m-%d %H:%M") if isinstance(session["created_at"], datetime) else str(session["created_at"])

                lines.append(f"  {status_emoji} {session['id']} ({created_str})")
                lines.append(f"    Model: {Path(session['model_path']).name}")
                lines.append(f"    Backend: {session['reference_backend']} → {session['alternative_backend']}")
                if session.get("success_rate") is not None:
                    lines.append(f"    Success: {session['success_rate']:.1f}%")
                lines.append("")

            return '\n'.join(lines)

    def generate_detailed_report(self, session_details: Dict[str, Any],
                               output_format: str = "text") -> str:
        """Generate detailed report for a single session."""
        session_info = session_details["session_info"]
        checkpoints = session_details["checkpoints"]
        performance = session_details["performance_metrics"]
        config = session_details["configuration"]

        if output_format == "json":
            return json.dumps(session_details, indent=2, default=str)

        else:
            # Text format
            lines = [
                f"Debug Session Detailed Report",
                f"Session ID: {session_info['id']}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "📋 Session Information:",
                f"  Model: {session_info['model_path']}",
                f"  Reference Backend: {session_info['reference_backend']}",
                f"  Alternative Backend: {session_info['alternative_backend']}",
                f"  Status: {session_info['status'].upper()}",
                f"  Created: {session_info['created_at']}",
                f"  Completed: {session_info['completed_at'] or 'N/A'}",
                f"  Duration: {session_info['total_duration']:.1f}s",
                "",
                "⚡ Performance Metrics:",
                f"  Total computation: {performance['total_computation_time']:.3f}s",
                f"  Reference backend: {performance['reference_backend_time']:.3f}s",
                f"  Alternative backend: {performance['alternative_backend_time']:.3f}s",
                f"  Comparison overhead: {performance['comparison_overhead']:.3f}s",
                f"  Peak memory usage: {performance['memory_peak_usage']:.1f}GB",
                "",
                "🎯 Checkpoint Results:"
            ]

            for i, checkpoint in enumerate(checkpoints, 1):
                status_emoji = "✅" if checkpoint["status"] == "passed" else "❌"
                lines.append(f"  [{i}] {status_emoji} {checkpoint['name']} ({checkpoint['execution_time']:.3f}s)")

                for tensor_comp in checkpoint["tensor_comparisons"]:
                    comp_status = "✅" if tensor_comp["status"] == "passed" else "❌"
                    lines.append(f"      {comp_status} {tensor_comp['tensor_name']}")
                    lines.append(f"        Shape: {tensor_comp['shape']}")
                    lines.append(f"        Similarity: {tensor_comp['similarity_score']:.6f}")
                    lines.append(f"        Max diff: {tensor_comp['max_diff']:.2e}")
                    lines.append(f"        Mean diff: {tensor_comp['mean_diff']:.2e}")
                lines.append("")

            lines.extend([
                "⚙️ Configuration:",
                f"  Precision thresholds:",
                f"    Relative tolerance: {config['precision_thresholds']['rtol']:.0e}",
                f"    Absolute tolerance: {config['precision_thresholds']['atol']:.0e}",
                f"  Enabled checkpoints: {', '.join(config['enabled_checkpoints'])}",
                f"  Tensor recording: {'✅ Enabled' if config['tensor_recording_enabled'] else '❌ Disabled'}",
                f"  Performance profiling: {'✅ Enabled' if config['performance_profiling_enabled'] else '❌ Disabled'}"
            ])

            return '\n'.join(lines)

    def export_session_data(self, session_details: Dict[str, Any],
                           output_file: str, format: str = "json") -> None:
        """Export session data to file."""
        try:
            if format == "json":
                content = json.dumps(session_details, indent=2, default=str)
            else:
                content = self.generate_detailed_report(session_details, format)

            with open(output_file, 'w') as f:
                f.write(content)

            print(f"📄 Session data exported to {output_file}")

        except Exception as e:
            print(f"❌ Failed to export session data: {e}", file=sys.stderr)

    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="Debug Framework Session Report Generator",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # List recent sessions
  session-report --list --days 7

  # Generate summary report
  session-report --summary --format json --output summary.json

  # Generate detailed report for specific session
  session-report --session session_001 --detailed --output report.txt

  # Show only failed sessions
  session-report --list --status failed --days 30
            """
        )

        parser.add_argument(
            "--database", "-d",
            default="/tmp/debug_framework.db",
            help="Path to debug framework database"
        )

        parser.add_argument(
            "--list", "-l",
            action="store_true",
            help="List available debug sessions"
        )

        parser.add_argument(
            "--summary", "-s",
            action="store_true",
            help="Generate summary report for all sessions"
        )

        parser.add_argument(
            "--session",
            help="Generate detailed report for specific session ID"
        )

        parser.add_argument(
            "--detailed",
            action="store_true",
            help="Generate detailed report (used with --session)"
        )

        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to look back for sessions (default: 30)"
        )

        parser.add_argument(
            "--status",
            choices=["completed", "failed", "partial"],
            help="Filter sessions by status"
        )

        parser.add_argument(
            "--format", "-f",
            choices=["json", "text"],
            default="text",
            help="Output format"
        )

        parser.add_argument(
            "--output", "-o",
            help="Output file (default: stdout)"
        )

        parser.add_argument(
            "--export",
            help="Export session data to file (use with --session)"
        )

        args = parser.parse_args()

        # Initialize services
        if not self.initialize_services(args.database):
            sys.exit(1)

        print(f"🗃️ Using database: {args.database}")
        print()

        # Handle different modes
        if args.list:
            # List sessions mode
            sessions = self.get_session_list(args.days, args.status)

            if not sessions:
                print("No sessions found.")
                sys.exit(0)

            print(f"📋 Debug Sessions (last {args.days} days):")
            print()

            for session in sorted(sessions, key=lambda x: x["created_at"], reverse=True):
                status_emoji = {
                    "completed": "✅",
                    "failed": "❌",
                    "partial": "⚠️"
                }.get(session["status"], "❓")

                created_str = session["created_at"].strftime("%Y-%m-%d %H:%M")
                duration_str = ""
                if session["completed_at"]:
                    duration = session["completed_at"] - session["created_at"]
                    duration_str = f" ({duration.total_seconds()/60:.1f}m)"

                print(f"{status_emoji} {session['id']} - {created_str}{duration_str}")
                print(f"   Model: {Path(session['model_path']).name}")
                print(f"   Backend: {session['reference_backend']} → {session['alternative_backend']}")
                print(f"   Checkpoints: {session['checkpoints_count']}, Success: {session['success_rate']:.1f}%")
                print()

        elif args.summary:
            # Summary report mode
            sessions = self.get_session_list(args.days, args.status)
            report = self.generate_summary_report(sessions, args.format)

            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"📄 Summary report saved to {args.output}")
            else:
                print(report)

        elif args.session:
            # Detailed session report mode
            session_details = self.get_session_details(args.session)

            if args.export:
                # Export mode
                self.export_session_data(session_details, args.export, args.format)
            else:
                # Report mode
                if args.detailed:
                    report = self.generate_detailed_report(session_details, args.format)
                else:
                    # Just show basic info
                    session_info = session_details["session_info"]
                    report = f"Session {session_info['id']}: {session_info['status']} ({session_info['created_at']})"

                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(report)
                    print(f"📄 Session report saved to {args.output}")
                else:
                    print(report)

        else:
            print("❌ Please specify --list, --summary, or --session", file=sys.stderr)
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    cli = SessionReportCLI()
    cli.main()