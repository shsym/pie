use bytes::Bytes;
use colored::*;
use prost::bytes;
use std::io;
use std::io::{IsTerminal, Write};
use uuid::Uuid;
use wasmtime::component::ResourceTable;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{
    IoView, OutputStream, Pollable, StdoutStream, StreamError, StreamResult, WasiCtx, WasiView,
};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

pub type Id = Uuid;

pub struct InstanceState {
    id: Id,
    arguments: Vec<String>,
    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,
}

impl IoView for InstanceState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
}

impl WasiView for InstanceState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi_ctx
    }
}

impl WasiHttpView for InstanceState {
    fn ctx(&mut self) -> &mut WasiHttpCtx {
        &mut self.http_ctx
    }
}

// Define your custom stdout type.

impl InstanceState {
    pub async fn new(id: Uuid, arguments: Vec<String>) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_network(); // TODO: Replace with socket_addr_check later.
        builder.stdout(Stdout(shorten_uuid(&id)));
        builder.stderr(Stderr(shorten_uuid(&id)));

        InstanceState {
            id,
            arguments,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
        }
    }

    pub fn id(&self) -> Id {
        self.id
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }
}

////////////////////////
// Helper functions for making stdout and stderr more readable.

fn shorten_uuid(uuid: &Uuid) -> String {
    // Convert the UUID to a string and split it by '-' to take the first segment.
    uuid.to_string().split('-').next().unwrap().to_string()
}

struct Stdout(String);
struct Stderr(String);

impl StdoutStream for Stdout {
    fn stream(&self) -> Box<dyn OutputStream> {
        Box::new(StdioOutputStream::Stdout(self.0.to_string()))
    }

    fn isatty(&self) -> bool {
        io::stderr().is_terminal()
    }
}

impl StdoutStream for Stderr {
    fn stream(&self) -> Box<dyn OutputStream> {
        Box::new(StdioOutputStream::Stderr(self.0.to_string()))
    }

    fn isatty(&self) -> bool {
        io::stderr().is_terminal()
    }
}

enum StdioOutputStream {
    Stdout(String),
    Stderr(String),
}

impl OutputStream for StdioOutputStream {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        // Convert the incoming bytes to a String (lossily, if needed)
        let message = String::from_utf8_lossy(&bytes);

        // Apply colors based on the stream type
        let colored_message = match self {
            StdioOutputStream::Stdout(prefix) => {
                format!("[Inst {}] {}", prefix, message).green().to_string()
            }
            StdioOutputStream::Stderr(prefix) => {
                format!("[Inst {}] {}", prefix, message).red().to_string()
            }
        };

        // Write the colored message to the appropriate output stream.
        match self {
            StdioOutputStream::Stdout(_) => io::stdout().write_all(colored_message.as_bytes()),
            StdioOutputStream::Stderr(_) => io::stderr().write_all(colored_message.as_bytes()),
        }
        .map_err(|e| StreamError::LastOperationFailed(anyhow::anyhow!(e)))
    }

    fn flush(&mut self) -> StreamResult<()> {
        match self {
            StdioOutputStream::Stdout(_) => io::stdout().flush(),
            StdioOutputStream::Stderr(_) => io::stderr().flush(),
        }
        .map_err(|e| StreamError::LastOperationFailed(anyhow::anyhow!(e)))
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        Ok(1024 * 1024)
    }
}

#[async_trait]
impl Pollable for StdioOutputStream {
    async fn ready(&mut self) {}
}
