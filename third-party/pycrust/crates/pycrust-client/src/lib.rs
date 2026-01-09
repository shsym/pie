//! PyCrust Client - Rust RPC client for Python workers
//!
//! This crate provides the client-side implementation for the PyCrust RPC framework.
//! It allows Rust applications to make asynchronous RPC calls to Python workers
//! using iceoryx2 shared memory for high-performance IPC.
//!
//! # Example
//!
//! ```ignore
//! use pycrust_client::RpcClient;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize)]
//! struct AddArgs { a: i32, b: i32 }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = RpcClient::connect("calculator").await?;
//!     let result: i32 = client.call("add", &AddArgs { a: 10, b: 20 }).await?;
//!     println!("Result: {}", result);
//!     client.close().await;
//!     Ok(())
//! }
//! ```

mod client;
mod error;
mod protocol;
mod transport;

pub use client::RpcClient;
pub use error::{Result, RpcError};
pub use protocol::{status, RpcRequest, RpcResponse};
