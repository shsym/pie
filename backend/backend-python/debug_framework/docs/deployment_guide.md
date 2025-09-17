# Multi-Backend Debug Framework - Deployment Guide

## Overview

This guide covers production deployment of the Multi-Backend Debug Framework for ML model validation across different backend implementations.

## Prerequisites

### System Requirements

- **Operating System**: Linux (recommended), macOS, or Windows
- **Python**: 3.11+

### Backend-Specific Requirements

#### Metal Backend (macOS only)
- macOS 12.0+ (Monterey or later)
- Apple Silicon (M1/M2/M3) or Intel Mac with Metal-compatible GPU
- Xcode Command Line Tools or Xcode 13.0+

#### CUDA Backend (Nvidia GPUs on Linux)
- CUDA Toolkit 11.0+ or 12.0+
- cuDNN 8.0+ (for optimized operations)

#### Python Backend
- PyTorch 1.12+ with appropriate backend support (CPU, CUDA, etc.)

## Installation

### Production Installation

```bash
# Create virtual environment
python -m venv pie-debug-framework
source pie-debug-framework/bin/activate  # Linux/macOS
# or
pie-debug-framework\Scripts\activate  # Windows

# Install from source with debug extras
cd backend/backend-python
pip install -e ".[debug]"

# Or using uv (recommended for faster installation)
uv sync --extra debug
```

### Docker Deployment

#### Dockerfile Example

```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements and install dependencies
COPY backend/backend-python/pyproject.toml .
COPY backend/backend-python/uv.lock .
RUN pip install uv && uv sync --extra debug

# Copy application code
COPY backend/backend-python/ .

# Create data directories
RUN mkdir -p /data/debug-framework/{database,logs,temp}

# Set environment variables
ENV PIE_DEBUG_DATABASE=/data/debug-framework/database/framework.db
ENV PIE_DEBUG_LEVEL=INFO
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import debug_framework; debug_framework.get_system_info()"

# Run as non-root user
RUN useradd --create-home --shell /bin/bash pie
RUN chown -R pie:pie /app /data
USER pie

EXPOSE 8080

CMD ["python", "-m", "debug_framework.cli.debug_validate", "--help"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  debug-framework:
    build: .
    environment:
      - PIE_DEBUG_ENABLED=true
      - PIE_DEBUG_LEVEL=INFO
      - PIE_DEBUG_DATABASE=/data/debug-framework/database/framework.db
    volumes:
      - debug_data:/data/debug-framework
      - model_data:/data/models:ro
    networks:
      - pie-network
    restart: unless-stopped

  debug-database:
    image: postgres:15
    environment:
      - POSTGRES_DB=debug_framework
      - POSTGRES_USER=debug_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - pie-network
    secrets:
      - db_password
    restart: unless-stopped

volumes:
  debug_data:
  model_data:
  postgres_data:

networks:
  pie-network:
    driver: bridge

secrets:
  db_password:
    external: true
```

## Configuration

### Environment Variables

#### Core Configuration
```bash
# Framework settings
export PIE_DEBUG_ENABLED=true
export PIE_DEBUG_LEVEL=INFO
export PIE_DEBUG_DATABASE=/opt/pie/debug/framework.db

# Backend paths
export PIE_METAL_PATH=/opt/pie/backends/metal
export PIE_CUDA_PATH=/opt/pie/backends/cuda

# Performance settings
export PIE_DEBUG_MAX_SESSIONS=100
export PIE_DEBUG_SESSION_TIMEOUT=3600
export PIE_DEBUG_TENSOR_LIMIT=1000

# Security settings
export PIE_DEBUG_SECURE_MODE=true
export PIE_DEBUG_LOG_LEVEL=WARNING
```

#### Production Configuration File

Create `/etc/pie/debug-framework.conf`:

```ini
[framework]
enabled = true
debug_level = INFO
database_path = /var/lib/pie/debug/framework.db
max_sessions = 100
session_timeout = 3600

[performance]
tensor_recording_limit = 1000
max_memory_usage_mb = 4096
enable_parallel_processing = true
max_concurrent_jobs = 4

[backends]
metal_path = /opt/pie/backends/metal
cuda_path = /opt/pie/backends/cuda
auto_detect_backends = true

[security]
secure_mode = true
restrict_file_access = true
audit_logging = true

[database]
connection_pool_size = 10
query_timeout = 30
auto_vacuum = true
backup_enabled = true
backup_interval_hours = 24
```

### Programmatic Configuration

```python
# config.py
PRODUCTION_CONFIG = {
    "database_path": "/var/lib/pie/debug/framework.db",
    "performance_monitoring": True,
    "auto_cleanup": True,
    "max_session_duration": 3600,
    "tensor_recording_limit": 1000,
    "comparison_tolerance": {
        "rtol": 1e-4,
        "atol": 1e-6
    },
    "security": {
        "secure_mode": True,
        "restrict_file_access": True,
        "audit_logging": True
    },
    "performance": {
        "max_memory_usage_mb": 4096,
        "enable_parallel_processing": True,
        "max_concurrent_jobs": 4
    }
}

# Initialize framework with production config
import debug_framework
debug_framework.initialize_framework(PRODUCTION_CONFIG)
```

## Database Setup

### SQLite (Default)

For single-server deployments:

```bash
# Create database directory
sudo mkdir -p /var/lib/pie/debug
sudo chown pie:pie /var/lib/pie/debug

# Initialize database (automatic on first use)
python -c "
import debug_framework
debug_framework.initialize_framework({
    'database_path': '/var/lib/pie/debug/framework.db'
})
"
```

### PostgreSQL (Recommended for Production)

For scalable, multi-server deployments:

```sql
-- Create database and user
CREATE DATABASE debug_framework;
CREATE USER debug_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE debug_framework TO debug_user;

-- Connect to database
\c debug_framework;

-- Create schema (handled automatically by framework)
-- Tables will be created on first framework initialization
```

Update configuration:

```python
POSTGRES_CONFIG = {
    "database_url": "postgresql://debug_user:secure_password@localhost:5432/debug_framework",
    "connection_pool_size": 10,
    "max_connections": 50
}
```

## Monitoring and Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name="debug_framework"):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_validation_event(self, event_type, session_id, details):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "details": details
        }
        self.logger.info(json.dumps(log_entry))

# Usage
logger = StructuredLogger()
logger.log_validation_event("validation_started", session_id, {
    "model_path": "/path/to/model.zt",
    "backends": ["python_reference", "metal"]
})
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
validation_counter = Counter('debug_framework_validations_total',
                           'Total validations performed', ['backend', 'status'])
validation_duration = Histogram('debug_framework_validation_duration_seconds',
                               'Validation duration', ['backend'])
active_sessions = Gauge('debug_framework_active_sessions',
                       'Number of active sessions')

class MetricsMiddleware:
    def __init__(self, validation_engine):
        self.validation_engine = validation_engine

    def execute_validation_with_metrics(self, session_id):
        start_time = time.time()
        active_sessions.inc()

        try:
            result = self.validation_engine.execute_validation(session_id)
            validation_counter.labels(
                backend=result.get('backend', 'unknown'),
                status='success'
            ).inc()
            return result
        except Exception as e:
            validation_counter.labels(
                backend='unknown',
                status='error'
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            validation_duration.labels(
                backend=result.get('backend', 'unknown')
            ).observe(duration)
            active_sessions.dec()
```

### Health Checks

```python
from flask import Flask, jsonify
import debug_framework

app = Flask(__name__)

@app.route('/health')
def health_check():
    try:
        # Check framework initialization
        system_info = debug_framework.get_system_info()

        # Check database connectivity
        from debug_framework.services.database_manager import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.get_connection()

        # Check backend availability
        backends = debug_framework.detect_available_backends()

        return jsonify({
            "status": "healthy",
            "framework_version": debug_framework.__version__,
            "available_backends": backends,
            "database_status": "connected"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    return generate_latest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Security

### Access Control

```python
import os
import stat
from pathlib import Path

class SecurityManager:
    def __init__(self):
        self.secure_mode = os.getenv('PIE_DEBUG_SECURE_MODE', 'true').lower() == 'true'

    def validate_file_access(self, file_path):
        """Validate file access permissions."""
        if not self.secure_mode:
            return True

        path = Path(file_path)

        # Check if file exists and is readable
        if not path.exists() or not os.access(path, os.R_OK):
            raise PermissionError(f"Cannot access file: {file_path}")

        # Restrict access to specific directories
        allowed_dirs = [
            "/opt/pie/models",
            "/var/lib/pie/debug",
            "/tmp/pie"
        ]

        if not any(path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
            raise PermissionError(f"File access restricted: {file_path}")

        return True

    def sanitize_session_config(self, config):
        """Sanitize session configuration."""
        safe_config = config.copy()

        # Remove potentially dangerous configuration
        dangerous_keys = ['system_commands', 'shell_access', 'file_operations']
        for key in dangerous_keys:
            safe_config.pop(key, None)

        # Limit resource usage
        safe_config['tensor_recording_limit'] = min(
            safe_config.get('tensor_recording_limit', 100), 1000
        )
        safe_config['max_session_duration'] = min(
            safe_config.get('max_session_duration', 3600), 7200
        )

        return safe_config

# Usage in validation engine
security_manager = SecurityManager()

def create_secure_validation_session(model_path, config):
    # Validate file access
    security_manager.validate_file_access(model_path)

    # Sanitize configuration
    safe_config = security_manager.sanitize_session_config(config)

    # Create session with sanitized config
    return create_validation_session(model_path, safe_config)
```

### Audit Logging

```python
import json
import hashlib
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file="/var/log/pie/debug-framework-audit.log"):
        self.log_file = log_file

    def log_event(self, event_type, user_id, resource, action, result):
        """Log security-relevant events."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,
            "checksum": None
        }

        # Add integrity checksum
        entry_str = json.dumps(audit_entry, sort_keys=True)
        audit_entry["checksum"] = hashlib.sha256(entry_str.encode()).hexdigest()

        # Write to audit log
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

# Usage
audit_logger = AuditLogger()
audit_logger.log_event(
    event_type="validation_session_created",
    user_id="debug_user",
    resource="model.zt",
    action="create_session",
    result="success"
)
```

## Performance Optimization

### Resource Management

```python
import psutil
import threading
from contextlib import contextmanager

class ResourceManager:
    def __init__(self, max_memory_mb=4096, max_cpu_percent=80):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self._active_sessions = {}
        self._lock = threading.Lock()

    @contextmanager
    def managed_session(self, session_id):
        """Context manager for resource-managed sessions."""
        try:
            self._acquire_resources(session_id)
            yield
        finally:
            self._release_resources(session_id)

    def _acquire_resources(self, session_id):
        with self._lock:
            # Check memory usage
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 90:
                raise ResourceError("System memory usage too high")

            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > self.max_cpu_percent:
                raise ResourceError("System CPU usage too high")

            # Track session resources
            self._active_sessions[session_id] = {
                "start_time": time.time(),
                "initial_memory": psutil.Process().memory_info().rss
            }

    def _release_resources(self, session_id):
        with self._lock:
            if session_id in self._active_sessions:
                session_info = self._active_sessions.pop(session_id)
                duration = time.time() - session_info["start_time"]

                # Log resource usage
                final_memory = psutil.Process().memory_info().rss
                memory_delta = final_memory - session_info["initial_memory"]

                logger.info(f"Session {session_id} completed: "
                           f"duration={duration:.2f}s, "
                           f"memory_delta={memory_delta/1024/1024:.2f}MB")

# Usage
resource_manager = ResourceManager()

with resource_manager.managed_session(session_id):
    validation_engine.execute_validation(session_id)
```

### Caching Strategy

```python
import redis
import pickle
from functools import wraps

class ValidationCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour

    def cache_key(self, model_path, backend, config):
        """Generate cache key for validation results."""
        import hashlib
        key_data = f"{model_path}:{backend}:{str(sorted(config.items()))}"
        return f"validation:{hashlib.md5(key_data.encode()).hexdigest()}"

    def get_cached_result(self, model_path, backend, config):
        """Get cached validation result."""
        cache_key = self.cache_key(model_path, backend, config)
        cached_data = self.redis_client.get(cache_key)

        if cached_data:
            return pickle.loads(cached_data)
        return None

    def cache_result(self, model_path, backend, config, result):
        """Cache validation result."""
        cache_key = self.cache_key(model_path, backend, config)
        serialized_result = pickle.dumps(result)
        self.redis_client.setex(cache_key, self.cache_ttl, serialized_result)

def cached_validation(cache):
    """Decorator for caching validation results."""
    def decorator(func):
        @wraps(func)
        def wrapper(model_path, backend, config):
            # Try cache first
            cached_result = cache.get_cached_result(model_path, backend, config)
            if cached_result:
                return cached_result

            # Execute validation
            result = func(model_path, backend, config)

            # Cache result
            cache.cache_result(model_path, backend, config, result)
            return result
        return wrapper
    return decorator

# Usage
validation_cache = ValidationCache()

@cached_validation(validation_cache)
def execute_validation_with_cache(model_path, backend, config):
    return validation_engine.execute_validation(session_id)
```

## High Availability

### Load Balancing

```python
import random
from typing import List, Dict

class ValidationLoadBalancer:
    def __init__(self, backend_endpoints: List[str]):
        self.backend_endpoints = backend_endpoints
        self.backend_health = {endpoint: True for endpoint in backend_endpoints}

    def get_healthy_backend(self) -> str:
        """Get a healthy backend endpoint using round-robin."""
        healthy_backends = [
            endpoint for endpoint, healthy in self.backend_health.items()
            if healthy
        ]

        if not healthy_backends:
            raise RuntimeError("No healthy backends available")

        return random.choice(healthy_backends)

    def check_backend_health(self, endpoint: str) -> bool:
        """Check if backend endpoint is healthy."""
        try:
            # Implement health check logic
            response = requests.get(f"{endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def update_backend_health(self):
        """Update health status for all backends."""
        for endpoint in self.backend_endpoints:
            self.backend_health[endpoint] = self.check_backend_health(endpoint)

# Usage
load_balancer = ValidationLoadBalancer([
    "http://debug-backend-1:8080",
    "http://debug-backend-2:8080",
    "http://debug-backend-3:8080"
])

# Periodically update health
import threading
import time

def health_check_worker():
    while True:
        load_balancer.update_backend_health()
        time.sleep(30)

health_thread = threading.Thread(target=health_check_worker, daemon=True)
health_thread.start()
```

### Failover Strategy

```python
class FailoverManager:
    def __init__(self, primary_backend, fallback_backends):
        self.primary_backend = primary_backend
        self.fallback_backends = fallback_backends

    def execute_with_failover(self, operation, *args, **kwargs):
        """Execute operation with automatic failover."""
        # Try primary backend first
        try:
            return operation(self.primary_backend, *args, **kwargs)
        except Exception as primary_error:
            logger.warning(f"Primary backend failed: {primary_error}")

            # Try fallback backends
            for fallback_backend in self.fallback_backends:
                try:
                    logger.info(f"Trying fallback backend: {fallback_backend}")
                    return operation(fallback_backend, *args, **kwargs)
                except Exception as fallback_error:
                    logger.warning(f"Fallback backend {fallback_backend} failed: {fallback_error}")
                    continue

            # All backends failed
            raise RuntimeError("All backends failed")

# Usage
failover_manager = FailoverManager(
    primary_backend="metal",
    fallback_backends=["cuda", "pytorch", "python_reference"]
)

result = failover_manager.execute_with_failover(
    lambda backend, model_path, config: execute_validation(model_path, backend, config),
    model_path="/path/to/model.zt",
    config=validation_config
)
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database file permissions
ls -la /var/lib/pie/debug/framework.db

# Test database connectivity
python -c "
from debug_framework.services.database_manager import DatabaseManager
db = DatabaseManager('/var/lib/pie/debug/framework.db')
print('Database connection successful')
"
```

#### Backend Detection Issues
```bash
# Check backend availability
python -c "
import debug_framework
backends = debug_framework.detect_available_backends()
print(f'Available backends: {backends}')
"

# Check Metal backend (macOS)
python -c "
from debug_framework.integrations.metal_backend import MetalBackend
metal = MetalBackend()
print(f'Metal initialization: {metal.initialize()}')
"
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
memory = psutil.virtual_memory()
print(f'Memory usage: {memory.percent}%')
print(f'Available: {memory.available / 1024 / 1024:.2f} MB')
"
```

### Debug Commands

```bash
# Enable debug logging
export PIE_DEBUG_ENABLED=true
export PIE_DEBUG_LEVEL=DEBUG

# Run validation with verbose output
python -m debug_framework.cli.debug_validate \
    --model-path /path/to/model.zt \
    --verbose \
    --output-file debug_results.json

# Check framework status
python -c "
import debug_framework
import json
info = debug_framework.get_system_info()
print(json.dumps(info, indent=2))
"
```

## Maintenance

### Database Maintenance

```bash
# Backup database
sqlite3 /var/lib/pie/debug/framework.db ".backup /backup/framework_$(date +%Y%m%d).db"

# Vacuum database (optimize space)
sqlite3 /var/lib/pie/debug/framework.db "VACUUM;"

# Analyze database statistics
sqlite3 /var/lib/pie/debug/framework.db "ANALYZE;"
```

### Log Rotation

```bash
# /etc/logrotate.d/pie-debug-framework
/var/log/pie/debug-framework*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        systemctl reload pie-debug-framework
    endscript
}
```

### Automated Cleanup

```python
import os
import time
from pathlib import Path

class CleanupManager:
    def __init__(self, database_path, max_age_days=30):
        self.database_path = database_path
        self.max_age_days = max_age_days

    def cleanup_old_sessions(self):
        """Remove sessions older than max_age_days."""
        from debug_framework.services.database_manager import DatabaseManager

        db_manager = DatabaseManager(self.database_path)
        cutoff_timestamp = time.time() - (self.max_age_days * 24 * 3600)

        query = """
        DELETE FROM debug_sessions
        WHERE created_at < ?
        AND status IN ('completed', 'failed')
        """

        db_manager.execute_query(query, (cutoff_timestamp,))

    def cleanup_temp_files(self):
        """Remove temporary files."""
        temp_dirs = ["/tmp/pie", "/var/tmp/pie"]

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for file_path in Path(temp_dir).rglob("*"):
                    if file_path.is_file() and \
                       time.time() - file_path.stat().st_mtime > 24 * 3600:
                        file_path.unlink()

# Cron job: 0 2 * * * /usr/local/bin/cleanup_debug_framework.py
```

This completes the comprehensive deployment guide for the Multi-Backend Debug Framework, covering all aspects of production deployment, monitoring, security, and maintenance.