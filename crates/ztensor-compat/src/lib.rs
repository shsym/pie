pub mod csr;

mod detect;
mod project;
mod safe;

#[cfg(feature = "safetensors")]
mod safetensors;

#[cfg(feature = "gguf")]
mod gguf;

#[cfg(feature = "npz")]
mod npz;

#[cfg(feature = "pickle")]
mod pt;

#[cfg(feature = "hdf5")]
mod hdf5;

#[cfg(feature = "onnx")]
mod onnx;

pub use detect::{detect, index, index_all, open, open_all, options, Open, FORMATS};
