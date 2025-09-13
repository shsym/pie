"""
Asynchronous logging system for performance optimization.

Provides non-blocking logging to minimize performance impact
during debug operations.
"""

import asyncio
import threading
import queue
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import time
from enum import Enum


class LogLevel(Enum):
    """Log levels for async logger."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Represents a log entry."""
    timestamp: float
    level: LogLevel
    message: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    operation_id: Optional[str] = None


class AsyncLogger:
    """
    High-performance asynchronous logger for debug framework.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        enable_console: bool = False
    ):
        """
        Initialize async logger.

        Args:
            log_file: Path to log file (None for in-memory only)
            max_queue_size: Maximum entries in queue before blocking
            batch_size: Number of entries to process in each batch
            flush_interval: Seconds between forced flushes
            enable_console: Whether to also log to console
        """
        self.log_file = log_file
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enable_console = enable_console

        # Thread-safe queue for log entries
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._running = threading.Event()
        self._running.set()

        # Background thread for processing
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Statistics
        self._stats = {
            'entries_logged': 0,
            'entries_dropped': 0,
            'batches_processed': 0,
            'flushes_performed': 0,
            'average_batch_time': 0.0
        }

        # Console logger if enabled
        self._console_logger = None
        if enable_console:
            self._console_logger = logging.getLogger('debug_framework_async')
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self._console_logger.addHandler(handler)
            self._console_logger.setLevel(logging.DEBUG)

    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        operation_id: Optional[str] = None
    ) -> bool:
        """
        Log a message asynchronously.

        Args:
            level: Log level
            message: Log message
            context: Additional context data
            session_id: Associated session ID
            operation_id: Associated operation ID

        Returns:
            True if logged successfully, False if queue is full
        """
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            context=context,
            session_id=session_id,
            operation_id=operation_id
        )

        try:
            self._queue.put_nowait(entry)
            self._stats['entries_logged'] += 1
            return True
        except queue.Full:
            self._stats['entries_dropped'] += 1
            return False

    def debug(self, message: str, **kwargs) -> bool:
        """Log debug message."""
        return self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> bool:
        """Log info message."""
        return self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> bool:
        """Log warning message."""
        return self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> bool:
        """Log error message."""
        return self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> bool:
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, message, **kwargs)

    def _worker_loop(self) -> None:
        """Background worker thread for processing log entries."""
        batch = []
        last_flush = time.time()

        while self._running.is_set():
            try:
                # Get entry with timeout
                try:
                    entry = self._queue.get(timeout=0.1)
                    batch.append(entry)
                except queue.Empty:
                    # Check if we should flush due to time
                    if batch and time.time() - last_flush >= self.flush_interval:
                        self._process_batch(batch)
                        batch = []
                        last_flush = time.time()
                    continue

                # Process batch when full or on timeout
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Exception as e:
                # Log error to console if available
                if self._console_logger:
                    self._console_logger.error(f"Error in async logger worker: {e}")

        # Process remaining entries before shutdown
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[LogEntry]) -> None:
        """Process a batch of log entries."""
        if not batch:
            return

        start_time = time.time()

        try:
            # Write to file if configured
            if self.log_file:
                self._write_to_file(batch)

            # Write to console if enabled
            if self._console_logger:
                self._write_to_console(batch)

            self._stats['batches_processed'] += 1
            self._stats['flushes_performed'] += 1

            # Update average batch processing time
            batch_time = time.time() - start_time
            current_avg = self._stats['average_batch_time']
            processed_batches = self._stats['batches_processed']

            self._stats['average_batch_time'] = (
                (current_avg * (processed_batches - 1) + batch_time) / processed_batches
            )

        except Exception as e:
            if self._console_logger:
                self._console_logger.error(f"Error processing log batch: {e}")

    def _write_to_file(self, batch: List[LogEntry]) -> None:
        """Write batch to log file."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'a', encoding='utf-8') as f:
            for entry in batch:
                log_data = {
                    'timestamp': entry.timestamp,
                    'level': entry.level.value,
                    'message': entry.message,
                    'session_id': entry.session_id,
                    'operation_id': entry.operation_id,
                    'context': entry.context
                }
                f.write(json.dumps(log_data) + '\n')

    def _write_to_console(self, batch: List[LogEntry]) -> None:
        """Write batch to console."""
        for entry in batch:
            # Map to standard logging levels
            level_map = {
                LogLevel.DEBUG: logging.DEBUG,
                LogLevel.INFO: logging.INFO,
                LogLevel.WARNING: logging.WARNING,
                LogLevel.ERROR: logging.ERROR,
                LogLevel.CRITICAL: logging.CRITICAL
            }

            log_level = level_map.get(entry.level, logging.INFO)

            # Format message with context
            message = entry.message
            if entry.session_id:
                message = f"[{entry.session_id}] {message}"
            if entry.operation_id:
                message = f"[{entry.operation_id}] {message}"
            if entry.context:
                context_str = json.dumps(entry.context)
                message = f"{message} | {context_str}"

            self._console_logger.log(log_level, message)

    def flush(self, timeout: float = 5.0) -> bool:
        """
        Force flush all pending log entries.

        Args:
            timeout: Maximum time to wait for flush

        Returns:
            True if successful, False if timeout
        """
        start_time = time.time()

        # Wait for queue to empty
        while not self._queue.empty():
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.01)

        return True

    def shutdown(self, timeout: float = 10.0) -> bool:
        """
        Shutdown async logger gracefully.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            True if shutdown successful
        """
        # Signal shutdown
        self._running.clear()

        # Wait for worker thread to finish
        self._worker_thread.join(timeout=timeout)

        return not self._worker_thread.is_alive()

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = self._stats.copy()
        stats.update({
            'queue_size': self._queue.qsize(),
            'queue_capacity': self.max_queue_size,
            'queue_utilization': self._queue.qsize() / self.max_queue_size,
            'worker_thread_alive': self._worker_thread.is_alive()
        })
        return stats

    def resize_queue(self, new_size: int) -> None:
        """
        Resize the internal queue (requires restart).

        Args:
            new_size: New maximum queue size
        """
        # This would require recreating the queue and worker thread
        # For simplicity, we'll just update the size for new instances
        self.max_queue_size = new_size


# Global async logger instance
_global_logger = None
_logger_lock = threading.Lock()


def get_async_logger() -> AsyncLogger:
    """Get global async logger instance (singleton)."""
    global _global_logger

    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                # Default configuration for debug framework
                log_file = "debug_framework/logs/debug_framework.log"
                _global_logger = AsyncLogger(
                    log_file=log_file,
                    max_queue_size=10000,
                    batch_size=50,
                    flush_interval=2.0,
                    enable_console=False  # Disable by default for performance
                )

    return _global_logger


def log_debug(message: str, **kwargs) -> bool:
    """Global debug log function."""
    return get_async_logger().debug(message, **kwargs)


def log_info(message: str, **kwargs) -> bool:
    """Global info log function."""
    return get_async_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs) -> bool:
    """Global warning log function."""
    return get_async_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs) -> bool:
    """Global error log function."""
    return get_async_logger().error(message, **kwargs)


def log_critical(message: str, **kwargs) -> bool:
    """Global critical log function."""
    return get_async_logger().critical(message, **kwargs)


def get_logging_stats() -> Dict[str, Any]:
    """Get global logging statistics."""
    return get_async_logger().get_stats()


def flush_logs(timeout: float = 5.0) -> bool:
    """Flush all pending logs."""
    return get_async_logger().flush(timeout)


def shutdown_logging(timeout: float = 10.0) -> bool:
    """Shutdown global logging system."""
    global _global_logger

    if _global_logger:
        result = _global_logger.shutdown(timeout)
        _global_logger = None
        return result

    return True