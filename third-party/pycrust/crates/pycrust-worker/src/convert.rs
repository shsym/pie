//! Conversion utilities between MessagePack and Python objects using pythonize.

use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};

/// Convert MessagePack bytes to a Python object.
///
/// Uses rmpv for MessagePack deserialization and pythonize for conversion to Python.
pub fn msgpack_to_pyobject(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    // First deserialize MessagePack to rmpv::Value
    // We use rmp_serde::from_slice directly to Value which implements Serialize,
    // allowing us to pipe it straight into pythonize.
    let value: rmpv::Value = rmp_serde::from_slice(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid msgpack: {}", e)))?;

    // Use pythonize to convert the rmpv::Value directly to a Python object.
    // rmpv::Value implements Serialize, so this works natively.
    // Note: rmpv::Value::Binary is serialized as bytes, so pythonize will create a bytes object.
    pythonize(py, &value)
        .map(|bound| bound.unbind())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Pythonize error: {}", e)))
}

/// Convert a Python object to MessagePack bytes.
///
/// Uses pythonize for Python to Rust conversion and rmp-serde for MessagePack serialization.
pub fn pyobject_to_msgpack(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    // Use depythonize to convert Python object to rmpv::Value directly
    let value: rmpv::Value = depythonize(obj)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Depythonize error: {}", e)))?;

    rmp_serde::to_vec_named(&value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyBytes, PyList};

    #[test]
    fn test_bytes_roundtrip() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let data = b"hello world";
            let py_bytes = PyBytes::new(py, data);
            
            // Python -> MsgPack
            let msgpack = pyobject_to_msgpack(py_bytes.as_any()).unwrap();
            
            // MsgPack -> Python
            let result = msgpack_to_pyobject(py, &msgpack).unwrap();
            
            assert!(result.bind(py).is_instance_of::<PyBytes>());
            let result_bytes: &[u8] = result.extract(py).unwrap();
            assert_eq!(result_bytes, data);
        });
    }

    #[test]
    fn test_list_roundtrip() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let data = vec![1, 2, 3];
            let py_list = PyList::new(py, &data).unwrap();
            
            // Python -> MsgPack
            let msgpack = pyobject_to_msgpack(py_list.as_any()).unwrap();
            
            // MsgPack -> Python
            let result = msgpack_to_pyobject(py, &msgpack).unwrap();
            
            assert!(result.bind(py).is_instance_of::<PyList>());
            let result_list: Vec<i32> = result.extract(py).unwrap();
            assert_eq!(result_list, data);
        });
    }
}
