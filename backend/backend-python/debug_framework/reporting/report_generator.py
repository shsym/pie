"""
Comprehensive report generation system for debug framework results.

Generates detailed reports for validation sessions, performance metrics,
and comprehensive debug summaries in multiple formats.
"""

import json
import csv
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
import sqlite3

from ..models.debug_session import DebugSession
from ..models.validation_report import ValidationReport
from ..models.tensor_comparison import TensorComparison
from ..models.tensor_recording import TensorRecording
from ..services.database_manager import DatabaseManager


class ReportGenerator:
    """
    Comprehensive report generator supporting multiple output formats
    and detailed analysis of debug framework results.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def generate_session_report(
        self,
        session_id: str,
        output_format: str = "html",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive report for a debug session.

        Args:
            session_id: ID of the debug session
            output_format: Format for the report (html, json, pdf, markdown)
            output_path: Optional path for output file

        Returns:
            Path to generated report file

        Raises:
            ValueError: If session not found or format not supported
        """
        session = self.db_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Gather all related data
        report_data = self._gather_session_data(session)

        # Generate report based on format
        if output_format.lower() == "html":
            return self._generate_html_report(report_data, output_path)
        elif output_format.lower() == "json":
            return self._generate_json_report(report_data, output_path)
        elif output_format.lower() == "markdown":
            return self._generate_markdown_report(report_data, output_path)
        elif output_format.lower() == "pdf":
            return self._generate_pdf_report(report_data, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _gather_session_data(self, session: DebugSession) -> Dict[str, Any]:
        """Gather all data related to a debug session."""
        data = {
            'session': asdict(session),
            'validation_reports': [],
            'tensor_comparisons': [],
            'tensor_recordings': [],
            'checkpoints': [],
            'performance_metrics': {},
            'summary': {}
        }

        try:
            # Get validation reports
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Validation reports
                cursor.execute(
                    "SELECT * FROM validation_reports WHERE session_id = ?",
                    (session.session_id,)
                )
                data['validation_reports'] = [dict(row) for row in cursor.fetchall()]

                # Tensor comparisons
                cursor.execute(
                    "SELECT * FROM tensor_comparisons WHERE session_id = ?",
                    (session.session_id,)
                )
                data['tensor_comparisons'] = [dict(row) for row in cursor.fetchall()]

                # Tensor recordings
                cursor.execute(
                    "SELECT * FROM tensor_recordings WHERE session_id = ?",
                    (session.session_id,)
                )
                data['tensor_recordings'] = [dict(row) for row in cursor.fetchall()]

                # Checkpoints
                cursor.execute(
                    "SELECT * FROM validation_checkpoints WHERE session_id = ?",
                    (session.session_id,)
                )
                data['checkpoints'] = [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error as e:
            print(f"Database error gathering session data: {e}")

        # Calculate summary statistics
        data['summary'] = self._calculate_summary_statistics(data)

        return data

    def _calculate_summary_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the session."""
        summary = {
            'total_validations': len(data['validation_reports']),
            'total_comparisons': len(data['tensor_comparisons']),
            'total_recordings': len(data['tensor_recordings']),
            'total_checkpoints': len(data['checkpoints']),
            'success_rate': 0.0,
            'average_error': 0.0,
            'max_error': 0.0,
            'min_error': float('inf'),
        }

        # Calculate success rate from validation reports
        if data['validation_reports']:
            successful = sum(1 for report in data['validation_reports']
                           if report.get('status') == 'passed')
            summary['success_rate'] = successful / len(data['validation_reports'])

        # Calculate error statistics from tensor comparisons
        if data['tensor_comparisons']:
            errors = [comp.get('max_error', 0.0) for comp in data['tensor_comparisons']
                     if comp.get('max_error') is not None]
            if errors:
                summary['average_error'] = sum(errors) / len(errors)
                summary['max_error'] = max(errors)
                summary['min_error'] = min(errors)

        return summary

    def _generate_html_report(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Generate HTML report."""
        session = data['session']
        summary = data['summary']

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debug Session Report - {session['session_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9e9e9; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Debug Session Report</h1>
        <p><strong>Session ID:</strong> {session['session_id']}</p>
        <p><strong>Backend:</strong> {session['backend']}</p>
        <p><strong>Status:</strong> <span class="{session['status'].lower()}">{session['status']}</span></p>
        <p><strong>Created:</strong> {session['created_at']}</p>
        <p><strong>Updated:</strong> {session['updated_at']}</p>
    </div>

    <div class="section">
        <h2>Summary Statistics</h2>
        <div class="metric">
            <strong>Success Rate:</strong> {summary['success_rate']:.2%}
        </div>
        <div class="metric">
            <strong>Total Validations:</strong> {summary['total_validations']}
        </div>
        <div class="metric">
            <strong>Total Comparisons:</strong> {summary['total_comparisons']}
        </div>
        <div class="metric">
            <strong>Average Error:</strong> {summary['average_error']:.2e}
        </div>
        <div class="metric">
            <strong>Max Error:</strong> {summary['max_error']:.2e}
        </div>
    </div>

    <div class="section">
        <h2>Validation Reports</h2>
        <table>
            <thead>
                <tr>
                    <th>Report ID</th>
                    <th>Operation</th>
                    <th>Status</th>
                    <th>Error Message</th>
                    <th>Created At</th>
                </tr>
            </thead>
            <tbody>
"""

        for report in data['validation_reports']:
            status_class = 'success' if report['status'] == 'passed' else 'failure'
            html_content += f"""
                <tr>
                    <td>{report['report_id']}</td>
                    <td>{report.get('operation', 'N/A')}</td>
                    <td class="{status_class}">{report['status']}</td>
                    <td>{report.get('error_message', '')}</td>
                    <td>{report['created_at']}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Tensor Comparisons</h2>
        <table>
            <thead>
                <tr>
                    <th>Comparison ID</th>
                    <th>Tensor A</th>
                    <th>Tensor B</th>
                    <th>Max Error</th>
                    <th>Mean Error</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

        for comp in data['tensor_comparisons']:
            status_class = 'success' if comp.get('status') == 'passed' else 'failure'
            html_content += f"""
                <tr>
                    <td>{comp['comparison_id']}</td>
                    <td>{comp.get('tensor_a_id', 'N/A')}</td>
                    <td>{comp.get('tensor_b_id', 'N/A')}</td>
                    <td>{comp.get('max_error', 0):.2e}</td>
                    <td>{comp.get('mean_error', 0):.2e}</td>
                    <td class="{status_class}">{comp.get('status', 'unknown')}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        # Write to file
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"debug_session_report_{session['session_id']}_{timestamp}.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_json_report(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Generate JSON report."""
        if not output_path:
            session_id = data['session']['session_id']
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"debug_session_report_{session_id}_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path

    def _generate_markdown_report(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Generate Markdown report."""
        session = data['session']
        summary = data['summary']

        md_content = f"""# Debug Session Report

## Session Information
- **Session ID**: {session['session_id']}
- **Backend**: {session['backend']}
- **Status**: {session['status']}
- **Created**: {session['created_at']}
- **Updated**: {session['updated_at']}

## Summary Statistics
- **Success Rate**: {summary['success_rate']:.2%}
- **Total Validations**: {summary['total_validations']}
- **Total Comparisons**: {summary['total_comparisons']}
- **Total Recordings**: {summary['total_recordings']}
- **Average Error**: {summary['average_error']:.2e}
- **Max Error**: {summary['max_error']:.2e}

## Validation Reports

| Report ID | Operation | Status | Error Message | Created At |
|-----------|-----------|--------|---------------|------------|
"""

        for report in data['validation_reports']:
            md_content += f"| {report['report_id']} | {report.get('operation', 'N/A')} | {report['status']} | {report.get('error_message', '')} | {report['created_at']} |\n"

        md_content += """
## Tensor Comparisons

| Comparison ID | Tensor A | Tensor B | Max Error | Mean Error | Status |
|---------------|----------|----------|-----------|------------|--------|
"""

        for comp in data['tensor_comparisons']:
            md_content += f"| {comp['comparison_id']} | {comp.get('tensor_a_id', 'N/A')} | {comp.get('tensor_b_id', 'N/A')} | {comp.get('max_error', 0):.2e} | {comp.get('mean_error', 0):.2e} | {comp.get('status', 'unknown')} |\n"

        if not output_path:
            session_id = data['session']['session_id']
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"debug_session_report_{session_id}_{timestamp}.md"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return output_path

    def _generate_pdf_report(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Generate PDF report (requires additional dependencies)."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
        except ImportError:
            # Fallback to HTML if reportlab not available
            html_path = self._generate_html_report(data, None)
            print(f"PDF generation requires reportlab. Generated HTML report instead: {html_path}")
            return html_path

        session = data['session']
        summary = data['summary']

        if not output_path:
            session_id = data['session']['session_id']
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"debug_session_report_{session_id}_{timestamp}.pdf"

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(f"Debug Session Report", title_style))
        story.append(Spacer(1, 12))

        # Session information
        story.append(Paragraph("Session Information", styles['Heading2']))
        session_info = [
            ["Session ID", session['session_id']],
            ["Backend", session['backend']],
            ["Status", session['status']],
            ["Created", session['created_at']],
            ["Updated", session['updated_at']]
        ]
        session_table = Table(session_info)
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(session_table)
        story.append(Spacer(1, 12))

        # Summary statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        summary_info = [
            ["Success Rate", f"{summary['success_rate']:.2%}"],
            ["Total Validations", str(summary['total_validations'])],
            ["Total Comparisons", str(summary['total_comparisons'])],
            ["Average Error", f"{summary['average_error']:.2e}"],
            ["Max Error", f"{summary['max_error']:.2e}"]
        ]
        summary_table = Table(summary_info)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)

        doc.build(story)
        return output_path

    def generate_performance_report(
        self,
        session_ids: List[str],
        output_format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate performance analysis report across multiple sessions.

        Args:
            session_ids: List of session IDs to analyze
            output_format: Format for the report (json, csv, html)
            output_path: Optional path for output file

        Returns:
            Path to generated report file
        """
        performance_data = []

        for session_id in session_ids:
            session_data = self._gather_session_data(
                self.db_manager.get_session(session_id)
            )

            perf_metrics = {
                'session_id': session_id,
                'backend': session_data['session']['backend'],
                'success_rate': session_data['summary']['success_rate'],
                'total_validations': session_data['summary']['total_validations'],
                'average_error': session_data['summary']['average_error'],
                'max_error': session_data['summary']['max_error'],
                'execution_time': session_data['session'].get('execution_time', 0)
            }
            performance_data.append(perf_metrics)

        if output_format.lower() == "csv":
            return self._generate_csv_performance_report(performance_data, output_path)
        elif output_format.lower() == "html":
            return self._generate_html_performance_report(performance_data, output_path)
        else:  # Default to JSON
            return self._generate_json_performance_report(performance_data, output_path)

    def _generate_csv_performance_report(self, data: List[Dict], output_path: Optional[str]) -> str:
        """Generate CSV performance report."""
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_report_{timestamp}.csv"

        if data:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

        return output_path

    def _generate_json_performance_report(self, data: List[Dict], output_path: Optional[str]) -> str:
        """Generate JSON performance report."""
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_report_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.datetime.now().isoformat(),
                'total_sessions': len(data),
                'performance_data': data
            }, f, indent=2)

        return output_path

    def _generate_html_performance_report(self, data: List[Dict], output_path: Optional[str]) -> str:
        """Generate HTML performance report."""
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_report_{timestamp}.html"

        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Analysis Report</h1>
        <p>Generated on: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p>Total Sessions: """ + str(len(data)) + """</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Session ID</th>
                <th>Backend</th>
                <th>Success Rate</th>
                <th>Total Validations</th>
                <th>Average Error</th>
                <th>Max Error</th>
                <th>Execution Time (s)</th>
            </tr>
        </thead>
        <tbody>
"""

        for row in data:
            html_content += f"""
            <tr>
                <td>{row['session_id']}</td>
                <td>{row['backend']}</td>
                <td>{row['success_rate']:.2%}</td>
                <td>{row['total_validations']}</td>
                <td>{row['average_error']:.2e}</td>
                <td>{row['max_error']:.2e}</td>
                <td>{row.get('execution_time', 0):.2f}</td>
            </tr>
"""

        html_content += """
        </tbody>
    </table>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path