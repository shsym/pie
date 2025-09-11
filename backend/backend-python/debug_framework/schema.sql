-- Multi-Backend Debugging Framework - SQLite Database Schema
-- Version: 1.0.0
-- Created: 2025-09-09
-- Description: Core database schema for session-centric debug validation system

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Debug Sessions Table
-- Stores high-level debugging session information with model context and configuration
CREATE TABLE IF NOT EXISTS debug_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    config TEXT CHECK(config IS NULL OR json_valid(config))
) STRICT;

-- Validation Checkpoints Table
-- Records validation checkpoints comparing backend implementations
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    checkpoint_name TEXT NOT NULL,
    reference_backend TEXT NOT NULL,
    alternative_backend TEXT NOT NULL,
    status TEXT CHECK (status IN ('pending', 'running', 'completed', 'failed')) DEFAULT 'pending',
    tensor_diff TEXT CHECK(tensor_diff IS NULL OR json_valid(tensor_diff)),
    execution_time_ms INTEGER CHECK(execution_time_ms IS NULL OR execution_time_ms >= 0),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, checkpoint_name),
    FOREIGN KEY (session_id) REFERENCES debug_sessions(id) ON DELETE CASCADE
) STRICT;

-- Tensor Recording Table
-- Stores individual tensor recordings with metadata and file path references
CREATE TABLE IF NOT EXISTS tensor_recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL,
    tensor_name TEXT NOT NULL,
    tensor_metadata TEXT CHECK(tensor_metadata IS NULL OR json_valid(tensor_metadata)),  -- {dtype, shape, stride, device, timestamp}
    tensor_data_path TEXT, -- Path to binary tensor file
    backend TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(checkpoint_id, backend, tensor_name),
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id) ON DELETE CASCADE
) STRICT;

-- Tensor Comparisons Table
-- Stores detailed tensor comparison results between reference and alternative backends
CREATE TABLE IF NOT EXISTS tensor_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL,
    tensor_name TEXT NOT NULL,
    shapes_match INTEGER NOT NULL CHECK(shapes_match IN (0, 1)),
    dtypes_match INTEGER NOT NULL CHECK(dtypes_match IN (0, 1)),
    max_absolute_diff REAL CHECK(max_absolute_diff IS NULL OR max_absolute_diff >= 0),
    max_relative_diff REAL CHECK(max_relative_diff IS NULL OR max_relative_diff >= 0),
    mean_absolute_error REAL CHECK(mean_absolute_error IS NULL OR mean_absolute_error >= 0),
    divergence_locations TEXT CHECK(divergence_locations IS NULL OR json_valid(divergence_locations)),
    statistical_summary TEXT CHECK(statistical_summary IS NULL OR json_valid(statistical_summary)),
    comparison_method TEXT CHECK(comparison_method IN ('element_wise', 'statistical', 'approximate')) DEFAULT 'element_wise',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id) ON DELETE CASCADE
) STRICT;

-- Indexes for Performance Optimization
-- Session queries by timestamp
CREATE INDEX IF NOT EXISTS idx_debug_sessions_created_at ON debug_sessions(created_at);

-- Checkpoint lookups by session and name
CREATE INDEX IF NOT EXISTS idx_checkpoints_session_id ON checkpoints(session_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON checkpoints(status);

-- Tensor recording lookups by checkpoint and name
CREATE INDEX IF NOT EXISTS idx_tensor_recordings_checkpoint_id ON tensor_recordings(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_tensor_recordings_tensor_name ON tensor_recordings(tensor_name);
CREATE INDEX IF NOT EXISTS idx_tensor_recordings_backend ON tensor_recordings(backend);

-- Tensor comparison lookups by checkpoint and tensor name
CREATE INDEX IF NOT EXISTS idx_tensor_comparisons_checkpoint_id ON tensor_comparisons(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_tensor_comparisons_tensor_name ON tensor_comparisons(tensor_name);
CREATE INDEX IF NOT EXISTS idx_tensor_comparisons_method ON tensor_comparisons(comparison_method);

-- Views for Common Queries
-- Session overview with checkpoint counts
CREATE VIEW IF NOT EXISTS session_overview AS
SELECT
    ds.id,
    ds.model_path,
    ds.created_at,
    COUNT(c.id) as checkpoint_count,
    COUNT(CASE WHEN c.status = 'completed' THEN 1 END) as completed_checkpoints,
    COUNT(CASE WHEN c.status = 'failed' THEN 1 END) as failed_checkpoints
FROM debug_sessions ds
LEFT JOIN checkpoints c ON ds.id = c.session_id
GROUP BY ds.id;

-- Checkpoint details with recording counts
CREATE VIEW IF NOT EXISTS checkpoint_details AS
SELECT
    c.id,
    c.session_id,
    c.checkpoint_name,
    c.reference_backend,
    c.alternative_backend,
    c.status,
    c.execution_time_ms,
    c.created_at,
    COUNT(tr.id) as recording_count
FROM checkpoints c
LEFT JOIN tensor_recordings tr ON c.id = tr.checkpoint_id
GROUP BY c.id;