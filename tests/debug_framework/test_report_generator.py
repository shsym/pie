"""
Tests for report generation system.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from debug_framework.reporting.report_generator import ReportGenerator
from debug_framework.models.debug_session import DebugSession
from debug_framework.services.database_manager import DatabaseManager


class TestReportGenerator:
    """Test cases for ReportGenerator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_db = Mock(spec=DatabaseManager)
        self.generator = ReportGenerator(self.mock_db)

    def test_init(self):
        """Test initialization of ReportGenerator."""
        assert self.generator.db_manager == self.mock_db

    @patch('debug_framework.reporting.report_generator.datetime')
    def test_generate_json_report(self, mock_datetime):
        """Test JSON report generation."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20240101_120000"

        # Mock session data
        session = DebugSession(
            session_id="test_session",
            backend="metal",
            status="completed"
        )

        self.mock_db.get_session.return_value = session

        # Mock database connection and data
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.side_effect = [
            [{'report_id': 'r1', 'status': 'passed', 'operation': 'softmax', 'created_at': '2024-01-01', 'error_message': None}],
            [{'comparison_id': 'c1', 'max_error': 0.001, 'mean_error': 0.0005, 'status': 'passed'}],
            [{'recording_id': 'rec1', 'tensor_name': 'input'}],
            [{'checkpoint_id': 'cp1', 'name': 'validation_point'}]
        ]

        self.mock_db.get_connection.return_value.__enter__.return_value = mock_conn

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_report.json")

            result_path = self.generator.generate_session_report(
                session_id="test_session",
                output_format="json",
                output_path=output_path
            )

            assert result_path == output_path
            assert os.path.exists(output_path)

            # Verify JSON content
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data['session']['session_id'] == 'test_session'
            assert data['session']['backend'] == 'metal'
            assert len(data['validation_reports']) == 1
            assert len(data['tensor_comparisons']) == 1

    def test_generate_session_report_not_found(self):
        """Test report generation for non-existent session."""
        self.mock_db.get_session.return_value = None

        with pytest.raises(ValueError, match="Session .* not found"):
            self.generator.generate_session_report("nonexistent", "json")

    def test_generate_session_report_unsupported_format(self):
        """Test report generation with unsupported format."""
        session = DebugSession(
            session_id="test_session",
            backend="metal",
            status="completed"
        )

        self.mock_db.get_session.return_value = session

        with pytest.raises(ValueError, match="Unsupported output format"):
            self.generator.generate_session_report("test_session", "xml")

    @patch('debug_framework.reporting.report_generator.datetime')
    def test_generate_html_report(self, mock_datetime):
        """Test HTML report generation."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20240101_120000"

        session = DebugSession(
            session_id="test_session",
            backend="metal",
            status="completed"
        )

        self.mock_db.get_session.return_value = session

        # Mock database data
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.side_effect = [
            [{'report_id': 'r1', 'status': 'passed', 'operation': 'softmax', 'created_at': '2024-01-01', 'error_message': None}],
            [{'comparison_id': 'c1', 'tensor_a_id': 'ta', 'tensor_b_id': 'tb', 'max_error': 0.001, 'mean_error': 0.0005, 'status': 'passed'}],
            [{'recording_id': 'rec1', 'tensor_name': 'input'}],
            [{'checkpoint_id': 'cp1', 'name': 'validation_point'}]
        ]

        self.mock_db.get_connection.return_value.__enter__.return_value = mock_conn

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_report.html")

            result_path = self.generator.generate_session_report(
                session_id="test_session",
                output_format="html",
                output_path=output_path
            )

            assert result_path == output_path
            assert os.path.exists(output_path)

            # Verify HTML contains expected content
            with open(output_path, 'r') as f:
                html_content = f.read()

            assert "test_session" in html_content
            assert "metal" in html_content
            assert "Summary Statistics" in html_content

    @patch('debug_framework.reporting.report_generator.datetime')
    def test_generate_markdown_report(self, mock_datetime):
        """Test Markdown report generation."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20240101_120000"

        session = DebugSession(
            session_id="test_session",
            backend="metal",
            status="completed"
        )

        self.mock_db.get_session.return_value = session

        # Mock database data
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.side_effect = [
            [{'report_id': 'r1', 'status': 'passed', 'operation': 'softmax', 'created_at': '2024-01-01', 'error_message': None}],
            [{'comparison_id': 'c1', 'tensor_a_id': 'ta', 'tensor_b_id': 'tb', 'max_error': 0.001, 'mean_error': 0.0005, 'status': 'passed'}],
            [{'recording_id': 'rec1', 'tensor_name': 'input'}],
            [{'checkpoint_id': 'cp1', 'name': 'validation_point'}]
        ]

        self.mock_db.get_connection.return_value.__enter__.return_value = mock_conn

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "test_report.md")

            result_path = self.generator.generate_session_report(
                session_id="test_session",
                output_format="markdown",
                output_path=output_path
            )

            assert result_path == output_path
            assert os.path.exists(output_path)

            # Verify Markdown content
            with open(output_path, 'r') as f:
                md_content = f.read()

            assert "# Debug Session Report" in md_content
            assert "test_session" in md_content
            assert "## Summary Statistics" in md_content

    def test_calculate_summary_statistics(self):
        """Test summary statistics calculation."""
        data = {
            'validation_reports': [
                {'status': 'passed'},
                {'status': 'passed'},
                {'status': 'failed'}
            ],
            'tensor_comparisons': [
                {'max_error': 0.001},
                {'max_error': 0.002},
                {'max_error': None}
            ],
            'tensor_recordings': [{}],
            'checkpoints': [{}, {}]
        }

        summary = self.generator._calculate_summary_statistics(data)

        assert summary['total_validations'] == 3
        assert summary['total_comparisons'] == 3
        assert summary['total_recordings'] == 1
        assert summary['total_checkpoints'] == 2
        assert summary['success_rate'] == 2/3  # 2 passed out of 3
        assert summary['average_error'] == 0.0015  # (0.001 + 0.002) / 2
        assert summary['max_error'] == 0.002
        assert summary['min_error'] == 0.001

    @patch('debug_framework.reporting.report_generator.datetime')
    def test_generate_performance_report_json(self, mock_datetime):
        """Test performance report generation in JSON format."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20240101_120000"
        mock_datetime.datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

        # Mock sessions for performance analysis
        session1 = DebugSession(session_id="s1", backend="metal", status="completed")
        session2 = DebugSession(session_id="s2", backend="cuda", status="completed")

        self.mock_db.get_session.side_effect = [session1, session2]

        # Mock database data for both sessions
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # First session data
        mock_cursor.fetchall.side_effect = [
            [{'report_id': 'r1', 'status': 'passed'}],  # validation_reports
            [{'comparison_id': 'c1', 'max_error': 0.001}],  # tensor_comparisons
            [],  # tensor_recordings
            [],  # checkpoints
            [{'report_id': 'r2', 'status': 'passed'}],  # Second session validation_reports
            [{'comparison_id': 'c2', 'max_error': 0.002}],  # Second session tensor_comparisons
            [],  # Second session tensor_recordings
            []   # Second session checkpoints
        ]

        self.mock_db.get_connection.return_value.__enter__.return_value = mock_conn

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "performance_report.json")

            result_path = self.generator.generate_performance_report(
                session_ids=["s1", "s2"],
                output_format="json",
                output_path=output_path
            )

            assert result_path == output_path
            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data['total_sessions'] == 2
            assert len(data['performance_data']) == 2

    @patch('debug_framework.reporting.report_generator.datetime')
    def test_generate_csv_performance_report(self, mock_datetime):
        """Test CSV performance report generation."""
        mock_datetime.datetime.now.return_value.strftime.return_value = "20240101_120000"

        data = [
            {
                'session_id': 's1',
                'backend': 'metal',
                'success_rate': 1.0,
                'total_validations': 10,
                'average_error': 0.001,
                'max_error': 0.002,
                'execution_time': 1.5
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "performance.csv")

            result_path = self.generator._generate_csv_performance_report(data, output_path)

            assert result_path == output_path
            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            assert 'session_id,backend,success_rate' in content
            assert 's1,metal,1.0' in content