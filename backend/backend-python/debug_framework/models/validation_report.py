"""
ValidationReport model for the debug framework.

This module defines the ValidationReport data model which provides
comprehensive summary of debugging session results including pass/fail status
and performance metrics.
"""

import json
import csv
import io
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from debug_framework.services import database_manager


class ValidationReport:
    """
    Comprehensive summary of debugging session results.

    Provides detailed analysis of validation outcomes, performance metrics,
    and formatted export capabilities for debugging session results.
    """

    # Valid report types (content types, not export formats)
    VALID_REPORT_TYPES = {"summary", "detailed"}

    # Valid overall statuses
    VALID_OVERALL_STATUSES = {"pass", "fail", "partial", "error"}

    # Valid checkpoint names for divergence points
    VALID_CHECKPOINT_NAMES = {
        "post_embedding", "post_rope", "post_attention",
        "pre_mlp", "post_mlp", "final_output"
    }

    # Required fields in performance summary
    REQUIRED_PERFORMANCE_FIELDS = {"reference_backend", "alternative_backend"}

    def __init__(
        self,
        session_id: int,
        report_type: str,
        overall_status: str,
        total_checkpoints: int,
        passed_checkpoints: int,
        failed_checkpoints: int,
        error_checkpoints: int,
        performance_summary: Dict[str, Any],
        report_content: str,
        skipped_checkpoints: int = 0,
        first_divergence_point: Optional[str] = None,
        id: Optional[int] = None,
        generated_at: Optional[datetime] = None
    ):
        """
        Initialize a ValidationReport instance.

        Args:
            session_id: ID of the associated debug session
            report_type: Type of report content (summary, detailed)
            overall_status: Overall validation result (pass, fail, partial, error)
            total_checkpoints: Total number of checkpoints in session
            passed_checkpoints: Number of checkpoints that passed validation
            failed_checkpoints: Number of checkpoints that failed validation
            error_checkpoints: Number of checkpoints that encountered errors
            performance_summary: Performance metrics comparing backends
            report_content: Main report content/analysis
            skipped_checkpoints: Number of checkpoints that were skipped
            first_divergence_point: First checkpoint where backends diverged
            id: Unique report identifier (auto-assigned if None)
            generated_at: Report generation timestamp (defaults to now)

        Raises:
            ValueError: If validation rules are violated
        """
        # Validate session ID
        if not isinstance(session_id, int) or session_id <= 0:
            raise ValueError("session_id must be a positive integer")

        # Validate report type
        self._validate_report_type(report_type)

        # Validate overall status
        self._validate_overall_status(overall_status)

        # Validate checkpoint counts
        self._validate_checkpoint_counts(
            total_checkpoints, passed_checkpoints, failed_checkpoints,
            error_checkpoints, skipped_checkpoints
        )

        # Validate status consistency with checkpoint outcomes
        self._validate_status_consistency(
            overall_status, passed_checkpoints, failed_checkpoints, error_checkpoints
        )

        # Validate performance summary
        self._validate_performance_summary(performance_summary)

        # Validate first divergence point if provided
        if first_divergence_point is not None:
            self._validate_first_divergence_point(first_divergence_point)

        # Validate required content
        if not report_content:
            raise ValueError("report_content is required and cannot be empty")

        # Store attributes
        self.id = id  # None until saved to database
        self.session_id = session_id
        self.report_type = report_type
        self.overall_status = overall_status
        self.total_checkpoints = total_checkpoints
        self.passed_checkpoints = passed_checkpoints
        self.failed_checkpoints = failed_checkpoints
        self.error_checkpoints = error_checkpoints
        self.skipped_checkpoints = skipped_checkpoints
        self.performance_summary = performance_summary.copy()  # Deep copy for safety
        self.report_content = report_content
        self.first_divergence_point = first_divergence_point
        self.generated_at = generated_at or datetime.now()

    def _validate_report_type(self, report_type: str) -> None:
        """Validate report type is valid."""
        if report_type not in self.VALID_REPORT_TYPES:
            valid_types = ", ".join(sorted(self.VALID_REPORT_TYPES))
            raise ValueError(f"Invalid report type '{report_type}'. Must be one of: {valid_types}")

    def _validate_overall_status(self, overall_status: str) -> None:
        """Validate overall status is valid."""
        if overall_status not in self.VALID_OVERALL_STATUSES:
            valid_statuses = ", ".join(sorted(self.VALID_OVERALL_STATUSES))
            raise ValueError(f"Invalid overall status '{overall_status}'. Must be one of: {valid_statuses}")

    def _validate_checkpoint_counts(
        self, total: int, passed: int, failed: int, error: int, skipped: int
    ) -> None:
        """Validate checkpoint counts are consistent."""
        # Check all counts are non-negative
        counts = [total, passed, failed, error, skipped]
        if any(count < 0 for count in counts):
            raise ValueError("All checkpoint counts must be non-negative")

        # Check total equals sum of components
        if total != passed + failed + error + skipped:
            raise ValueError(
                "total_checkpoints must equal passed_checkpoints + failed_checkpoints + error_checkpoints + skipped_checkpoints"
            )

    def _validate_status_consistency(
        self, overall_status: str, passed: int, failed: int, error: int
    ) -> None:
        """Validate overall status reflects checkpoint outcomes accurately."""
        if overall_status == "pass":
            # Pass requires no failures or errors
            if failed > 0 or error > 0:
                raise ValueError("overall_status must reflect checkpoint outcomes accurately")
        elif overall_status == "error":
            # Error status requires at least one error
            if error == 0:
                raise ValueError("overall_status must reflect checkpoint outcomes accurately")
        # Note: "fail" and "partial" can have various combinations, so less strict validation

    def _validate_performance_summary(self, performance_summary: Dict[str, Any]) -> None:
        """Validate performance summary contains required fields."""
        if not isinstance(performance_summary, dict):
            raise ValueError("performance_summary must be a dictionary")

        # Check for required fields - both reference_backend and alternative_backend must be present
        missing_fields = self.REQUIRED_PERFORMANCE_FIELDS - set(performance_summary.keys())
        if missing_fields:
            missing_names = ", ".join(sorted(missing_fields))
            raise ValueError(f"performance_summary must include reference and alternative timing (missing: {missing_names})")

    def _validate_first_divergence_point(self, divergence_point: str) -> None:
        """Validate first divergence point is a valid checkpoint name."""
        if divergence_point not in self.VALID_CHECKPOINT_NAMES:
            valid_names = ", ".join(sorted(self.VALID_CHECKPOINT_NAMES))
            raise ValueError(f"first_divergence_point must reference valid checkpoint name. Valid names: {valid_names}")

    @classmethod
    def generate_from_session(cls, session_id: int, report_type: str = "summary") -> 'ValidationReport':
        """
        Generate validation report from debug session data.

        Args:
            session_id: ID of debug session to generate report for
            report_type: Type of report to generate (summary or detailed)

        Returns:
            ValidationReport instance generated from session data

        Note:
            This method imports DebugSession dynamically to avoid circular imports.
        """
        from debug_framework.models.debug_session import DebugSession

        session = DebugSession.load(session_id)

        # TODO: Implement actual checkpoint analysis
        # For now, create a basic report structure
        performance_summary = {
            "reference_backend": {"total_time_ms": 100},
            "alternative_backend": {"total_time_ms": 90},
            "speedup_factor": 1.11
        }

        return cls(
            session_id=session_id,
            report_type=report_type,
            overall_status="pass",  # Default - would be calculated from actual checkpoints
            total_checkpoints=1,
            passed_checkpoints=1,
            failed_checkpoints=0,
            error_checkpoints=0,
            skipped_checkpoints=0,
            performance_summary=performance_summary,
            report_content=f"Generated {report_type} report for session {session_id}",
            first_divergence_point=None
        )

    def export_json(self, file_path: str) -> None:
        """
        Export report to JSON file.

        Args:
            file_path: Path to write JSON file
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def export_csv(self, file_path: str) -> None:
        """
        Export report summary to CSV file.

        Args:
            file_path: Path to write CSV file
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow([
                'session_id', 'report_type', 'overall_status', 'total_checkpoints',
                'passed_checkpoints', 'failed_checkpoints', 'error_checkpoints',
                'skipped_checkpoints', 'first_divergence_point', 'generated_at'
            ])

            # Write data row
            writer.writerow([
                self.session_id, self.report_type, self.overall_status, self.total_checkpoints,
                self.passed_checkpoints, self.failed_checkpoints, self.error_checkpoints,
                self.skipped_checkpoints, self.first_divergence_point,
                self.generated_at.isoformat()
            ])

    def export_text(self, file_path: str) -> None:
        """
        Export report to plain text file.

        Args:
            file_path: Path to write text file
        """
        with open(file_path, 'w') as f:
            f.write(self.to_text())

    def export_html(self, file_path: str) -> None:
        """
        Export report to HTML file.

        Args:
            file_path: Path to write HTML file
        """
        with open(file_path, 'w') as f:
            f.write(self.to_html())

    def to_text(self) -> str:
        """
        Generate plain text representation of report.

        Returns:
            Plain text formatted report
        """
        lines = [
            f"Validation Report - Session {self.session_id}",
            "=" * 50,
            f"Report Type: {self.report_type.title()}",
            f"Overall Status: {self.overall_status.upper()}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Checkpoint Summary:",
            f"  Total: {self.total_checkpoints}",
            f"  Passed: {self.passed_checkpoints}",
            f"  Failed: {self.failed_checkpoints}",
            f"  Errors: {self.error_checkpoints}",
            f"  Skipped: {self.skipped_checkpoints}",
            ""
        ]

        if self.first_divergence_point:
            lines.extend([
                f"First Divergence: {self.first_divergence_point}",
                ""
            ])

        # Add performance summary
        if self.performance_summary:
            lines.extend([
                "Performance Summary:",
                f"  Reference Backend: {self.performance_summary.get('reference_backend', {}).get('total_time_ms', 'N/A')}ms",
                f"  Alternative Backend: {self.performance_summary.get('alternative_backend', {}).get('total_time_ms', 'N/A')}ms",
                ""
            ])

        lines.extend([
            "Report Content:",
            "-" * 20,
            self.report_content
        ])

        return "\n".join(lines)

    def to_html(self) -> str:
        """
        Generate HTML representation of report.

        Returns:
            HTML formatted report
        """
        status_color = {
            "pass": "#28a745",
            "fail": "#dc3545",
            "partial": "#ffc107",
            "error": "#6c757d"
        }.get(self.overall_status, "#6c757d")

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Validation Report - Session {self.session_id}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }",
            ".status { font-weight: bold; font-size: 1.2em; }",
            ".summary { margin: 20px 0; }",
            ".content { background-color: #ffffff; padding: 20px; border: 1px solid #dee2e6; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            '<div class="header">',
            f"<h1>Validation Report - Session {self.session_id}</h1>",
            f'<div class="status" style="color: {status_color}">Status: {self.overall_status.upper()}</div>',
            f"<p>Report Type: {self.report_type.title()}</p>",
            f"<p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "</div>",
            '<div class="summary">',
            "<h2>Checkpoint Summary</h2>",
            "<ul>",
            f"<li>Total: {self.total_checkpoints}</li>",
            f"<li>Passed: {self.passed_checkpoints}</li>",
            f"<li>Failed: {self.failed_checkpoints}</li>",
            f"<li>Errors: {self.error_checkpoints}</li>",
            f"<li>Skipped: {self.skipped_checkpoints}</li>",
            "</ul>"
        ]

        if self.first_divergence_point:
            html_parts.extend([
                f"<p><strong>First Divergence:</strong> {self.first_divergence_point}</p>"
            ])

        html_parts.extend([
            "</div>",
            '<div class="content">',
            "<h2>Report Content</h2>",
            f"<p>{self.report_content}</p>",
            "</div>",
            "</body>",
            "</html>"
        ])

        return "\n".join(html_parts)

    def get_success_rate(self) -> float:
        """
        Calculate success rate as percentage of passed checkpoints.

        Returns:
            Success rate as percentage (0.0 to 100.0)
        """
        if self.total_checkpoints == 0:
            return 0.0

        # Only count non-skipped checkpoints for success rate
        executed_checkpoints = self.total_checkpoints - self.skipped_checkpoints
        if executed_checkpoints == 0:
            return 0.0

        return (self.passed_checkpoints / executed_checkpoints) * 100.0

    def get_performance_improvement(self) -> Optional[float]:
        """
        Calculate performance improvement factor.

        Returns:
            Performance improvement factor, or None if data unavailable
        """
        ref_time = self.performance_summary.get("reference_backend", {}).get("total_time_ms")
        alt_time = self.performance_summary.get("alternative_backend", {}).get("total_time_ms")

        if ref_time and alt_time and ref_time > 0:
            return ref_time / alt_time

        return None

    def has_divergence(self) -> bool:
        """
        Check if report indicates backend divergence.

        Returns:
            True if backends diverged, False otherwise
        """
        return self.first_divergence_point is not None

    def save(self) -> int:
        """
        Save validation report to database.

        Returns:
            Database ID of the saved report
        """
        db_manager = database_manager.DatabaseManager()

        # Prepare data for database insertion
        report_data = {
            "session_id": self.session_id,
            "report_type": self.report_type,
            "overall_status": self.overall_status,
            "total_checkpoints": self.total_checkpoints,
            "passed_checkpoints": self.passed_checkpoints,
            "failed_checkpoints": self.failed_checkpoints,
            "error_checkpoints": self.error_checkpoints,
            "skipped_checkpoints": self.skipped_checkpoints,
            "performance_summary": json.dumps(self.performance_summary),
            "report_content": self.report_content,
            "first_divergence_point": self.first_divergence_point,
            "generated_at": self.generated_at.isoformat()
        }

        if self.id is None:
            # Insert new report
            self.id = db_manager.insert_validation_report(report_data)
        else:
            # Update existing report
            db_manager.update_validation_report(self.id, report_data)

        return self.id

    @classmethod
    def load(cls, report_id: int) -> 'ValidationReport':
        """
        Load validation report from database by ID.

        Args:
            report_id: Database ID of the report

        Returns:
            ValidationReport instance loaded from database

        Raises:
            ValueError: If report not found in database
        """
        db_manager = database_manager.DatabaseManager()
        report_data = db_manager.get_validation_report(report_id)

        if not report_data:
            raise ValueError(f"Validation report with ID {report_id} not found")

        return cls.from_dict(report_data)

    @classmethod
    def find_by_session(cls, session_id: int) -> List['ValidationReport']:
        """
        Find all validation reports for a session.

        Args:
            session_id: ID of debug session

        Returns:
            List of ValidationReport instances for the session
        """
        db_manager = database_manager.DatabaseManager()
        reports_data = db_manager.get_validation_reports_by_session(session_id)

        return [cls.from_dict(report_data) for report_data in reports_data]

    def to_dict(self) -> Dict[str, Any]:
        """Convert ValidationReport to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "report_type": self.report_type,
            "overall_status": self.overall_status,
            "total_checkpoints": self.total_checkpoints,
            "passed_checkpoints": self.passed_checkpoints,
            "failed_checkpoints": self.failed_checkpoints,
            "error_checkpoints": self.error_checkpoints,
            "skipped_checkpoints": self.skipped_checkpoints,
            "performance_summary": self.performance_summary,
            "report_content": self.report_content,
            "first_divergence_point": self.first_divergence_point,
            "generated_at": self.generated_at.isoformat()
        }

    def to_json(self) -> str:
        """Convert ValidationReport to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationReport':
        """Create ValidationReport from dictionary representation."""
        # Parse JSON fields
        performance_summary = json.loads(data.get("performance_summary", "{}")) if isinstance(data.get("performance_summary"), str) else data.get("performance_summary", {})

        # Parse timestamp field
        generated_at = None
        if data.get("generated_at"):
            if isinstance(data["generated_at"], str):
                try:
                    generated_at = datetime.fromisoformat(data["generated_at"].replace('Z', '+00:00'))
                except ValueError:
                    generated_at = datetime.now()  # Default to now if parsing fails
            elif isinstance(data["generated_at"], datetime):
                generated_at = data["generated_at"]

        return cls(
            id=data.get("id"),
            session_id=data["session_id"],
            report_type=data["report_type"],
            overall_status=data["overall_status"],
            total_checkpoints=data["total_checkpoints"],
            passed_checkpoints=data["passed_checkpoints"],
            failed_checkpoints=data["failed_checkpoints"],
            error_checkpoints=data["error_checkpoints"],
            skipped_checkpoints=data.get("skipped_checkpoints", 0),
            performance_summary=performance_summary,
            report_content=data["report_content"],
            first_divergence_point=data.get("first_divergence_point"),
            generated_at=generated_at
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ValidationReport':
        """Create ValidationReport from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of ValidationReport."""
        return (f"ValidationReport(id={self.id}, session_id={self.session_id}, "
                f"status='{self.overall_status}', checkpoints={self.passed_checkpoints}/{self.total_checkpoints})")

    def get_detailed_content(self) -> str:
        """
        Get detailed content of the validation report.

        Returns:
            Detailed report content with extended information
        """
        detailed_lines = [
            f"=== DETAILED VALIDATION REPORT ===",
            f"Session ID: {self.session_id}",
            f"Report Type: {self.report_type}",
            f"Overall Status: {self.overall_status}",
            f"Generated: {self.generated_at}",
            "",
            "=== CHECKPOINT SUMMARY ===",
            f"Total Checkpoints: {self.total_checkpoints}",
            f"Passed: {self.passed_checkpoints}",
            f"Failed: {self.failed_checkpoints}",
            f"Errors: {self.error_checkpoints}",
            f"Skipped: {self.skipped_checkpoints}",
            "",
        ]

        if self.first_divergence_point:
            detailed_lines.extend([
                "=== DIVERGENCE ANALYSIS ===",
                f"First Divergence Point: {self.first_divergence_point}",
                "",
            ])

        if self.performance_summary:
            detailed_lines.extend([
                "=== PERFORMANCE ANALYSIS ===",
                f"Reference Backend Total Time: {self.performance_summary.get('reference_backend', {}).get('total_time_ms', 'N/A')}ms",
                f"Alternative Backend Total Time: {self.performance_summary.get('alternative_backend', {}).get('total_time_ms', 'N/A')}ms",
                "",
            ])

        detailed_lines.extend([
            "=== FULL REPORT CONTENT ===",
            self.report_content,
            "",
            "=== END DETAILED REPORT ==="
        ])

        return "\n".join(detailed_lines)

    def calculate_overall_speedup(self) -> Optional[float]:
        """
        Calculate overall speedup between reference and alternative backends.

        Returns:
            Speedup ratio (reference_time / alternative_time) or None if not available
        """
        if not self.performance_summary:
            return None

        ref_time = self.performance_summary.get('reference_backend', {}).get('total_time_ms')
        alt_time = self.performance_summary.get('alternative_backend', {}).get('total_time_ms')

        if ref_time is None or alt_time is None or alt_time == 0:
            return None

        return float(ref_time) / float(alt_time)

    def contains_keyword(self, keyword: str) -> bool:
        """
        Check if the report contains a specific keyword.

        Args:
            keyword: Keyword to search for

        Returns:
            True if keyword is found in report content, False otherwise
        """
        if not keyword:
            return False

        keyword_lower = keyword.lower()

        # Search in report content
        if keyword_lower in self.report_content.lower():
            return True

        # Search in first divergence point
        if self.first_divergence_point and keyword_lower in self.first_divergence_point.lower():
            return True

        return False

    def calculate_checkpoint_speedups(self) -> Dict[str, float]:
        """
        Calculate speedup factors for individual checkpoints.

        Returns:
            Dictionary mapping checkpoint names to speedup factors
        """
        speedups = {}

        if not self.performance_summary:
            return speedups

        ref_checkpoint_times = self.performance_summary.get('reference_backend', {}).get('checkpoint_times', {})
        alt_checkpoint_times = self.performance_summary.get('alternative_backend', {}).get('checkpoint_times', {})

        # Calculate speedup for each checkpoint that has timing data
        for checkpoint_name in ref_checkpoint_times:
            if checkpoint_name in alt_checkpoint_times:
                ref_time = ref_checkpoint_times[checkpoint_name]
                alt_time = alt_checkpoint_times[checkpoint_name]

                if alt_time > 0:
                    speedups[checkpoint_name] = ref_time / alt_time

        return speedups

    def matches_status(self, status_filter: str) -> bool:
        """
        Check if the report matches a status filter.

        Args:
            status_filter: Status to match against

        Returns:
            True if report status matches the filter
        """
        return self.overall_status.lower() == status_filter.lower()

    def has_failures(self) -> bool:
        """
        Check if the report has any failed checkpoints.

        Returns:
            True if there are failed checkpoints
        """
        return self.failed_checkpoints > 0

    def has_errors(self) -> bool:
        """
        Check if the report has any error checkpoints.

        Returns:
            True if there are error checkpoints
        """
        return self.error_checkpoints > 0

    def __repr__(self) -> str:
        """Detailed representation of ValidationReport."""
        return (f"ValidationReport(id={self.id}, session_id={self.session_id}, "
                f"report_type='{self.report_type}', overall_status='{self.overall_status}', "
                f"total_checkpoints={self.total_checkpoints})")