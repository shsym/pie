//! PyCrust Worker - Python extension for RPC workers
//!
//! This module provides the PyO3 bindings for running PyCrust workers.
//! Workers listen for incoming RPC requests from Rust clients and dispatch
//! them to registered Python handlers.

mod convert;
mod worker;

use pyo3::prelude::*;

/// Python module definition for pycrust._worker
#[pymodule]
#[pyo3(name = "_worker")]
pub fn pycrust_worker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(worker::run_worker, m)?)?;
    Ok(())
}
