"""
Test module for ValidationReport model.

This test module validates the ValidationReport data model which provides
comprehensive summary of debugging session results including pass/fail status and performance metrics.

TDD: This test MUST FAIL until the ValidationReport model is implemented.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.models.validation_report import ValidationReport
    VALIDATIONREPORT_AVAILABLE = True
except ImportError:
    ValidationReport = None
    VALIDATIONREPORT_AVAILABLE = False


class TestValidationReport:
    """Test suite for ValidationReport model functionality."""

    def test_validation_report_import_fails(self):
        """Test that ValidationReport import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.models.validation_report import ValidationReport

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_validation_report_creation(self):
        """Test basic ValidationReport object creation."""
        performance_summary = {
            "reference_backend": {
                "total_time_ms": 150,
                "checkpoint_times": {"post_attention": 80, "post_mlp": 70}
            },
            "alternative_backend": {
                "total_time_ms": 120,
                "checkpoint_times": {"post_attention": 60, "post_mlp": 60}
            },
            "speedup_factor": 1.25
        }

        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=5,
            passed_checkpoints=3,
            failed_checkpoints=1,
            error_checkpoints=0,
            skipped_checkpoints=1,
            performance_summary=performance_summary,
            first_divergence_point="post_mlp",
            report_content="Validation completed with minor differences in MLP layer."
        )

        assert report.session_id == 1
        assert report.report_type == "summary"
        assert report.overall_status == "pass"
        assert report.total_checkpoints == 5
        assert report.passed_checkpoints == 3
        assert report.failed_checkpoints == 1
        assert report.error_checkpoints == 0
        assert report.skipped_checkpoints == 1
        assert report.performance_summary == performance_summary
        assert report.first_divergence_point == "post_mlp"
        assert report.generated_at is not None

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_report_type_validation(self):
        """Test report type validation."""
        valid_types = ["summary", "detailed"]  # Removed "html" - that's an export format
        performance_summary = {"reference_backend": {"total_time_ms": 100}}

        # Test valid report types
        for report_type in valid_types:
            report = ValidationReport(
                session_id=1,
                report_type=report_type,
                overall_status="pass",
                total_checkpoints=1,
                passed_checkpoints=1,
                failed_checkpoints=0,
                error_checkpoints=0,
                performance_summary=performance_summary,
                report_content="Test report"
            )
            assert report.report_type == report_type

        # Test invalid report type
        with pytest.raises(ValueError, match="Invalid report type"):
            ValidationReport(
                session_id=1,
                report_type="html",  # HTML is export format, not content type
                overall_status="pass",
                total_checkpoints=1,
                passed_checkpoints=1,
                failed_checkpoints=0,
                error_checkpoints=0,
                performance_summary=performance_summary,
                report_content="Test report"
            )

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_overall_status_validation(self):
        """Test overall status validation."""
        valid_statuses = ["pass", "fail", "partial", "error"]
        performance_summary = {"reference_backend": {"total_time_ms": 100}}

        # Test valid statuses
        for status in valid_statuses:
            report = ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status=status,
                total_checkpoints=1,
                passed_checkpoints=1 if status == "pass" else 0,
                failed_checkpoints=0 if status == "pass" else 1,
                error_checkpoints=0,
                performance_summary=performance_summary,
                report_content="Test report"
            )
            assert report.overall_status == status

        # Test invalid status
        with pytest.raises(ValueError, match="Invalid overall status"):
            ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status="invalid_status",
                total_checkpoints=1,
                passed_checkpoints=1,
                failed_checkpoints=0,
                error_checkpoints=0,
                performance_summary=performance_summary,
                report_content="Test report"
            )

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_checkpoint_count_validation(self):
        """Test checkpoint count consistency validation."""
        performance_summary = {"reference_backend": {"total_time_ms": 100}}

        # Valid checkpoint counts
        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="partial",
            total_checkpoints=10,
            passed_checkpoints=6,
            failed_checkpoints=2,
            error_checkpoints=1,
            skipped_checkpoints=1,
            performance_summary=performance_summary,
            report_content="Test report"
        )

        # Should validate that total = passed + failed + error + skipped
        assert report.total_checkpoints == 10
        assert report.passed_checkpoints + report.failed_checkpoints + report.error_checkpoints + report.skipped_checkpoints == 10

        # Invalid checkpoint counts (don't sum to total)
        with pytest.raises(ValueError, match="total_checkpoints must equal passed_checkpoints \\+ failed_checkpoints \\+ error_checkpoints \\+ skipped_checkpoints"):
            ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status="pass",
                total_checkpoints=10,
                passed_checkpoints=5,
                failed_checkpoints=3,
                error_checkpoints=1,
                skipped_checkpoints=0,  # 5 + 3 + 1 + 0 = 9, not 10
                performance_summary=performance_summary,
                report_content="Test report"
            )

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_overall_status_consistency(self):
        """Test overall status reflects checkpoint outcomes accurately."""
        performance_summary = {"reference_backend": {"total_time_ms": 100}}

        # Test "pass" status requires all non-skipped checkpoints passed
        report_pass = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=5,
            passed_checkpoints=4,
            failed_checkpoints=0,
            error_checkpoints=0,
            skipped_checkpoints=1,  # Skipped checkpoints don't affect pass status
            performance_summary=performance_summary,
            report_content="All executed checkpoints passed"
        )
        assert report_pass.overall_status == "pass"

        # Test "fail" status when any checkpoints failed
        report_fail = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="fail",
            total_checkpoints=5,
            passed_checkpoints=2,
            failed_checkpoints=2,
            error_checkpoints=0,
            skipped_checkpoints=1,
            performance_summary=performance_summary,
            report_content="Some checkpoints failed"
        )
        assert report_fail.overall_status == "fail"

        # Test "error" status when any checkpoints errored
        report_error = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="error",
            total_checkpoints=5,
            passed_checkpoints=2,
            failed_checkpoints=1,
            error_checkpoints=1,
            skipped_checkpoints=1,
            performance_summary=performance_summary,
            report_content="Some checkpoints errored"
        )
        assert report_error.overall_status == "error"

        # Test invalid status/count combination
        with pytest.raises(ValueError, match="overall_status must reflect checkpoint outcomes accurately"):
            ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status="pass",  # Claims pass
                total_checkpoints=5,
                passed_checkpoints=2,
                failed_checkpoints=2,  # But has failures
                error_checkpoints=0,
                skipped_checkpoints=1,
                performance_summary=performance_summary,
                report_content="Invalid status"
            )

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_performance_summary_validation(self):
        """Test performance summary must include reference and alternative timing."""
        # Valid performance summary
        valid_summary = {
            "reference_backend": {
                "total_time_ms": 150,
                "checkpoint_times": {"post_attention": 80, "post_mlp": 70},
                "average_time_per_checkpoint": 75
            },
            "alternative_backend": {
                "total_time_ms": 120,
                "checkpoint_times": {"post_attention": 60, "post_mlp": 60},
                "average_time_per_checkpoint": 60
            },
            "speedup_factor": 1.25,
            "performance_improvement": "20% faster"
        }

        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=2,
            passed_checkpoints=2,
            failed_checkpoints=0,
            error_checkpoints=0,
            performance_summary=valid_summary,
            report_content="Performance test"
        )
        assert report.performance_summary == valid_summary

        # Invalid performance summary missing required fields
        invalid_summary = {
            "reference_backend": {"total_time_ms": 150},
            # Missing alternative_backend
        }

        with pytest.raises(ValueError, match="performance_summary must include reference and alternative timing"):
            ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status="pass",
                total_checkpoints=1,
                passed_checkpoints=1,
                failed_checkpoints=0,
                error_checkpoints=0,
                performance_summary=invalid_summary,
                report_content="Invalid performance"
            )

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_first_divergence_point_validation(self):
        """Test first divergence point must reference valid checkpoint name."""
        performance_summary = {"reference_backend": {"total_time_ms": 100}, "alternative_backend": {"total_time_ms": 90}}
        valid_checkpoint_names = ["post_embedding", "post_rope", "post_attention", "pre_mlp", "post_mlp", "final_output"]

        # Test valid checkpoint names
        for checkpoint in valid_checkpoint_names:
            report = ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status="fail",
                total_checkpoints=1,
                passed_checkpoints=0,
                failed_checkpoints=1,
                error_checkpoints=0,
                performance_summary=performance_summary,
                first_divergence_point=checkpoint,
                report_content="Test divergence"
            )
            assert report.first_divergence_point == checkpoint

        # Test invalid checkpoint name
        with pytest.raises(ValueError, match="first_divergence_point must reference valid checkpoint name"):
            ValidationReport(
                session_id=1,
                report_type="summary",
                overall_status="fail",
                total_checkpoints=1,
                passed_checkpoints=0,
                failed_checkpoints=1,
                error_checkpoints=0,
                performance_summary=performance_summary,
                first_divergence_point="invalid_checkpoint",
                report_content="Invalid divergence"
            )

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_report_generation_from_session(self):
        """Test report generation from DebugSession data."""
        # Mock session with checkpoints
        with patch('debug_framework.models.debug_session.DebugSession') as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.id = 1
            mock_session_instance.status = "completed"
            mock_session.load.return_value = mock_session_instance


            # Test report generation
            report = ValidationReport.generate_from_session(session_id=1, report_type="summary")

            assert report.session_id == 1
            assert isinstance(report.total_checkpoints, int)
            assert isinstance(report.performance_summary, dict)

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_html_export_format(self):
        """Test HTML export format (separate from content type)."""
        performance_summary = {
            "reference_backend": {"total_time_ms": 150},
            "alternative_backend": {"total_time_ms": 120},
            "speedup_factor": 1.25
        }

        report = ValidationReport(
            session_id=1,
            report_type="detailed",  # Content type is detailed
            overall_status="pass",
            total_checkpoints=3,
            passed_checkpoints=3,
            failed_checkpoints=0,
            error_checkpoints=0,
            skipped_checkpoints=0,
            performance_summary=performance_summary,
            report_content="Detailed report with comprehensive analysis"
        )

        # Test HTML export (format concern)
        with patch('builtins.open', mock_open()) as mock_file:
            report.export_html("/path/to/report.html")
            mock_file.assert_called_once_with("/path/to/report.html", 'w')

        # Test getting HTML representation of detailed content
        html_content = report.to_html()
        assert "<html>" in html_content
        assert "Detailed report" in html_content

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_detailed_report_content(self):
        """Test detailed report content generation."""
        performance_summary = {
            "reference_backend": {"total_time_ms": 150},
            "alternative_backend": {"total_time_ms": 120}
        }

        # Mock detailed checkpoint data
        with patch('debug_framework.models.validation_checkpoint.ValidationCheckpoint') as mock_checkpoint:
            mock_checkpoint.get_by_session.return_value = [
                MagicMock(checkpoint_name="post_attention", comparison_status="pass"),
                MagicMock(checkpoint_name="post_mlp", comparison_status="fail")
            ]

            report = ValidationReport(
                session_id=1,
                report_type="detailed",
                overall_status="partial",
                total_checkpoints=2,
                passed_checkpoints=1,
                failed_checkpoints=1,
                error_checkpoints=0,
                performance_summary=performance_summary,
                report_content="Detailed analysis with checkpoint breakdown"
            )

            # Test detailed content includes checkpoint information
            detailed_content = report.get_detailed_content()
            assert "checkpoint breakdown" in detailed_content.lower()
            assert report.report_type == "detailed"

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_database_integration(self):
        """Test database persistence operations."""
        performance_summary = {
            "reference_backend": {"total_time_ms": 150},
            "alternative_backend": {"total_time_ms": 120}
        }

        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=3,
            passed_checkpoints=3,
            failed_checkpoints=0,
            error_checkpoints=0,
            performance_summary=performance_summary,
            report_content="Test report content"
        )

        # Mock database operations
        with patch('debug_framework.services.database_manager.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance

            # Test save operation
            report.save()
            mock_db_instance.insert_validation_report.assert_called_once()

            # Test load operation
            mock_db_instance.get_validation_report.return_value = {
                'id': 1,
                'session_id': 1,
                'report_type': 'summary',
                'overall_status': 'pass',
                'total_checkpoints': 3,
                'passed_checkpoints': 3,
                'failed_checkpoints': 0,
                'error_checkpoints': 0,
                'performance_summary': json.dumps(performance_summary),
                'first_divergence_point': None,
                'report_content': 'Test report content',
                'generated_at': datetime.now().isoformat()
            }

            loaded_report = ValidationReport.load(1)
            assert loaded_report.session_id == 1
            assert loaded_report.overall_status == "pass"

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        performance_summary = {
            "reference_backend": {"total_time_ms": 150, "checkpoint_times": {"post_attention": 80}},
            "alternative_backend": {"total_time_ms": 120, "checkpoint_times": {"post_attention": 60}},
            "speedup_factor": 1.25
        }

        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=1,
            passed_checkpoints=1,
            failed_checkpoints=0,
            error_checkpoints=0,
            skipped_checkpoints=0,
            performance_summary=performance_summary,
            first_divergence_point=None,
            report_content="JSON test report"
        )

        # Test serialization
        report_dict = report.to_dict()
        assert report_dict["session_id"] == 1
        assert report_dict["report_type"] == "summary"
        assert report_dict["performance_summary"] == performance_summary

        # Test deserialization
        restored_report = ValidationReport.from_dict(report_dict)
        assert restored_report.session_id == report.session_id
        assert restored_report.performance_summary == report.performance_summary

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_performance_analysis(self):
        """Test performance analysis calculations."""
        performance_summary = {
            "reference_backend": {
                "total_time_ms": 200,
                "checkpoint_times": {"post_attention": 120, "post_mlp": 80}
            },
            "alternative_backend": {
                "total_time_ms": 150,
                "checkpoint_times": {"post_attention": 90, "post_mlp": 60}
            }
        }

        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=2,
            passed_checkpoints=2,
            failed_checkpoints=0,
            error_checkpoints=0,
            performance_summary=performance_summary,
            report_content="Performance analysis"
        )

        # Test performance calculations
        speedup = report.calculate_overall_speedup()
        assert speedup == 200 / 150  # 1.33x speedup

        checkpoint_speedups = report.calculate_checkpoint_speedups()
        assert checkpoint_speedups["post_attention"] == 120 / 90
        assert checkpoint_speedups["post_mlp"] == 80 / 60

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_report_filtering_and_search(self):
        """Test report filtering and search capabilities."""
        performance_summary = {"reference_backend": {"total_time_ms": 100}, "alternative_backend": {"total_time_ms": 90}}

        report = ValidationReport(
            session_id=1,
            report_type="detailed",
            overall_status="partial",
            total_checkpoints=5,
            passed_checkpoints=3,
            failed_checkpoints=2,
            error_checkpoints=0,
            performance_summary=performance_summary,
            first_divergence_point="post_mlp",
            report_content="Detailed report with post_mlp checkpoint failure and attention success"
        )

        # Test content search
        assert report.contains_keyword("post_mlp") is True
        assert report.contains_keyword("attention") is True
        assert report.contains_keyword("nonexistent") is False

        # Test status filtering
        assert report.matches_status("partial") is True
        assert report.matches_status("pass") is False

        # Test checkpoint filtering
        assert report.has_failures() is True
        assert report.has_errors() is False

    @pytest.mark.skipif(not VALIDATIONREPORT_AVAILABLE, reason="ValidationReport not implemented")
    def test_report_export_formats(self):
        """Test different report export formats."""
        performance_summary = {"reference_backend": {"total_time_ms": 100}, "alternative_backend": {"total_time_ms": 90}}

        report = ValidationReport(
            session_id=1,
            report_type="summary",
            overall_status="pass",
            total_checkpoints=2,
            passed_checkpoints=2,
            failed_checkpoints=0,
            error_checkpoints=0,
            performance_summary=performance_summary,
            report_content="Export format test"
        )

        # Test JSON export
        with patch('builtins.open', mock_open()) as mock_file:
            report.export_json("/path/to/report.json")
            mock_file.assert_called_with("/path/to/report.json", 'w')

        # Test CSV export (summary data)
        with patch('builtins.open', mock_open()) as mock_file:
            report.export_csv("/path/to/report.csv")
            mock_file.assert_called_with("/path/to/report.csv", 'w')

        # Test text export
        with patch('builtins.open', mock_open()) as mock_file:
            report.export_text("/path/to/report.txt")
            mock_file.assert_called_with("/path/to/report.txt", 'w')