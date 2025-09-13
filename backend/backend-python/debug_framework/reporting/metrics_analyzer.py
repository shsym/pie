"""
Metrics analysis utilities for debug framework reports.

Provides statistical analysis and trend detection for debug framework metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import sqlite3
from dataclasses import dataclass

from ..services.database_manager import DatabaseManager


@dataclass
class MetricsTrend:
    """Represents a trend in metrics over time."""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    slope: float
    r_squared: float
    confidence: float


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data."""
    backend: str
    operation: str
    mean_execution_time: float
    std_execution_time: float
    success_rate: float
    error_rate: float


class MetricsAnalyzer:
    """
    Statistical analysis and trend detection for debug framework metrics.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def analyze_session_trends(
        self,
        backend: str,
        time_window_days: int = 30
    ) -> List[MetricsTrend]:
        """
        Analyze trends in session metrics over time.

        Args:
            backend: Backend to analyze
            time_window_days: Number of days to look back

        Returns:
            List of detected trends
        """
        trends = []

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get session data over time window
            cursor.execute("""
                SELECT created_at, session_id
                FROM debug_sessions
                WHERE backend = ?
                AND datetime(created_at) >= datetime('now', '-' || ? || ' days')
                ORDER BY created_at
            """, (backend, time_window_days))

            sessions = cursor.fetchall()

            if len(sessions) < 3:  # Need minimum data points
                return trends

            # Analyze success rate trend
            success_trend = self._analyze_success_rate_trend(sessions)
            if success_trend:
                trends.append(success_trend)

            # Analyze error rate trend
            error_trend = self._analyze_error_trend(sessions)
            if error_trend:
                trends.append(error_trend)

            # Analyze execution time trend
            exec_time_trend = self._analyze_execution_time_trend(sessions)
            if exec_time_trend:
                trends.append(exec_time_trend)

        return trends

    def _analyze_success_rate_trend(self, sessions: List[Tuple]) -> Optional[MetricsTrend]:
        """Analyze success rate trend over time."""
        success_rates = []
        timestamps = []

        for i, (created_at, session_id) in enumerate(sessions):
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed
                    FROM validation_reports
                    WHERE session_id = ?
                """, (session_id,))

                result = cursor.fetchone()
                if result and result[0] > 0:
                    success_rate = result[1] / result[0]
                    success_rates.append(success_rate)
                    timestamps.append(i)

        if len(success_rates) < 3:
            return None

        return self._calculate_trend("success_rate", timestamps, success_rates)

    def _analyze_error_trend(self, sessions: List[Tuple]) -> Optional[MetricsTrend]:
        """Analyze error trend over time."""
        avg_errors = []
        timestamps = []

        for i, (created_at, session_id) in enumerate(sessions):
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT AVG(max_error)
                    FROM tensor_comparisons
                    WHERE session_id = ? AND max_error IS NOT NULL
                """, (session_id,))

                result = cursor.fetchone()
                if result and result[0] is not None:
                    avg_errors.append(result[0])
                    timestamps.append(i)

        if len(avg_errors) < 3:
            return None

        return self._calculate_trend("average_error", timestamps, avg_errors)

    def _analyze_execution_time_trend(self, sessions: List[Tuple]) -> Optional[MetricsTrend]:
        """Analyze execution time trend over time."""
        exec_times = []
        timestamps = []

        for i, (created_at, session_id) in enumerate(sessions):
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT execution_time
                    FROM debug_sessions
                    WHERE session_id = ? AND execution_time IS NOT NULL
                """, (session_id,))

                result = cursor.fetchone()
                if result and result[0] is not None:
                    exec_times.append(result[0])
                    timestamps.append(i)

        if len(exec_times) < 3:
            return None

        return self._calculate_trend("execution_time", timestamps, exec_times)

    def _calculate_trend(
        self,
        metric_name: str,
        x_values: List[float],
        y_values: List[float]
    ) -> Optional[MetricsTrend]:
        """Calculate linear trend for given data points."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return None

        try:
            # Calculate linear regression
            x_array = np.array(x_values, dtype=float)
            y_array = np.array(y_values, dtype=float)

            slope, intercept = np.polyfit(x_array, y_array, 1)

            # Calculate R-squared
            y_pred = slope * x_array + intercept
            ss_res = np.sum((y_array - y_pred) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Determine trend direction
            if abs(slope) < 0.001:  # Threshold for "stable"
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            # Calculate confidence based on R-squared and data points
            confidence = min(r_squared * (len(x_values) / 10), 1.0)

            return MetricsTrend(
                metric_name=metric_name,
                trend_direction=direction,
                slope=float(slope),
                r_squared=float(r_squared),
                confidence=float(confidence)
            )

        except Exception as e:
            print(f"Error calculating trend for {metric_name}: {e}")
            return None

    def generate_performance_benchmarks(
        self,
        backends: List[str] = None
    ) -> List[PerformanceBenchmark]:
        """
        Generate performance benchmarks for backends.

        Args:
            backends: List of backends to analyze (None for all)

        Returns:
            List of performance benchmarks
        """
        benchmarks = []

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Build query based on backends filter
            if backends:
                backend_filter = f"WHERE backend IN ({','.join(['?'] * len(backends))})"
                query_params = backends
            else:
                backend_filter = ""
                query_params = []

            # Get all sessions for analysis
            cursor.execute(f"""
                SELECT backend, session_id, execution_time
                FROM debug_sessions
                {backend_filter}
                ORDER BY backend
            """, query_params)

            sessions = cursor.fetchall()

            # Group by backend
            backend_sessions = defaultdict(list)
            for backend, session_id, exec_time in sessions:
                backend_sessions[backend].append((session_id, exec_time))

            # Calculate benchmarks for each backend
            for backend, session_data in backend_sessions.items():
                benchmark = self._calculate_backend_benchmark(backend, session_data)
                if benchmark:
                    benchmarks.append(benchmark)

        return benchmarks

    def _calculate_backend_benchmark(
        self,
        backend: str,
        session_data: List[Tuple]
    ) -> Optional[PerformanceBenchmark]:
        """Calculate performance benchmark for a single backend."""
        execution_times = []
        total_validations = 0
        total_passed = 0
        total_errors = 0

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            for session_id, exec_time in session_data:
                if exec_time is not None:
                    execution_times.append(exec_time)

                # Get validation statistics
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed
                    FROM validation_reports
                    WHERE session_id = ?
                """, (session_id,))

                result = cursor.fetchone()
                if result:
                    total_validations += result[0] or 0
                    total_passed += result[1] or 0

                # Get error count
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM tensor_comparisons
                    WHERE session_id = ? AND max_error > 0
                """, (session_id,))

                error_result = cursor.fetchone()
                if error_result:
                    total_errors += error_result[0] or 0

        if not execution_times:
            return None

        # Calculate statistics
        mean_exec_time = np.mean(execution_times)
        std_exec_time = np.std(execution_times)
        success_rate = total_passed / total_validations if total_validations > 0 else 0.0
        error_rate = total_errors / total_validations if total_validations > 0 else 0.0

        return PerformanceBenchmark(
            backend=backend,
            operation="overall",  # Could be made more specific
            mean_execution_time=float(mean_exec_time),
            std_execution_time=float(std_exec_time),
            success_rate=float(success_rate),
            error_rate=float(error_rate)
        )

    def detect_anomalies(
        self,
        session_id: str,
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in session metrics compared to historical data.

        Args:
            session_id: Session to analyze for anomalies
            threshold_std: Number of standard deviations for anomaly threshold

        Returns:
            List of detected anomalies
        """
        anomalies = []

        session = self.db_manager.get_session(session_id)
        if not session:
            return anomalies

        backend = session.backend

        # Get historical statistics for comparison
        historical_stats = self._get_historical_statistics(backend)

        # Check current session metrics against historical data
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check success rate anomaly
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed
                FROM validation_reports
                WHERE session_id = ?
            """, (session_id,))

            result = cursor.fetchone()
            if result and result[0] > 0:
                current_success_rate = result[1] / result[0]
                if self._is_anomalous(
                    current_success_rate,
                    historical_stats.get('success_rate', {}),
                    threshold_std
                ):
                    anomalies.append({
                        'metric': 'success_rate',
                        'current_value': current_success_rate,
                        'historical_mean': historical_stats['success_rate'].get('mean', 0),
                        'severity': 'high' if abs(current_success_rate - historical_stats['success_rate'].get('mean', 0)) > threshold_std * 2 else 'medium'
                    })

            # Check error rate anomaly
            cursor.execute("""
                SELECT AVG(max_error)
                FROM tensor_comparisons
                WHERE session_id = ? AND max_error IS NOT NULL
            """, (session_id,))

            error_result = cursor.fetchone()
            if error_result and error_result[0] is not None:
                current_avg_error = error_result[0]
                if self._is_anomalous(
                    current_avg_error,
                    historical_stats.get('avg_error', {}),
                    threshold_std
                ):
                    anomalies.append({
                        'metric': 'average_error',
                        'current_value': current_avg_error,
                        'historical_mean': historical_stats['avg_error'].get('mean', 0),
                        'severity': 'high' if current_avg_error > historical_stats['avg_error'].get('mean', 0) + threshold_std * 2 else 'medium'
                    })

        return anomalies

    def _get_historical_statistics(self, backend: str, days_back: int = 30) -> Dict[str, Dict[str, float]]:
        """Get historical statistics for a backend."""
        stats = {}

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get sessions from the time window
            cursor.execute("""
                SELECT session_id
                FROM debug_sessions
                WHERE backend = ?
                AND datetime(created_at) >= datetime('now', '-' || ? || ' days')
            """, (backend, days_back))

            session_ids = [row[0] for row in cursor.fetchall()]

            if not session_ids:
                return stats

            # Calculate success rate statistics
            success_rates = []
            avg_errors = []

            for session_id in session_ids:
                # Success rate
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed
                    FROM validation_reports
                    WHERE session_id = ?
                """, (session_id,))

                result = cursor.fetchone()
                if result and result[0] > 0:
                    success_rates.append(result[1] / result[0])

                # Average error
                cursor.execute("""
                    SELECT AVG(max_error)
                    FROM tensor_comparisons
                    WHERE session_id = ? AND max_error IS NOT NULL
                """, (session_id,))

                error_result = cursor.fetchone()
                if error_result and error_result[0] is not None:
                    avg_errors.append(error_result[0])

            # Calculate statistics
            if success_rates:
                stats['success_rate'] = {
                    'mean': np.mean(success_rates),
                    'std': np.std(success_rates)
                }

            if avg_errors:
                stats['avg_error'] = {
                    'mean': np.mean(avg_errors),
                    'std': np.std(avg_errors)
                }

        return stats

    def _is_anomalous(
        self,
        current_value: float,
        historical_stats: Dict[str, float],
        threshold_std: float
    ) -> bool:
        """Check if current value is anomalous compared to historical data."""
        if not historical_stats or 'mean' not in historical_stats or 'std' not in historical_stats:
            return False

        mean = historical_stats['mean']
        std = historical_stats['std']

        if std == 0:  # No variation in historical data
            return current_value != mean

        z_score = abs(current_value - mean) / std
        return z_score > threshold_std