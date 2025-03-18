use anyhow::Result;
use futures::{SinkExt, StreamExt};
use rmp_serde::{decode, encode};

use crate::server::{ClientMessage, ServerMessage};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

const CHUNK_SIZE: usize = 64 * 1024;

/// A client that interacts with the "Symphony" server.
pub struct Client {
    /// Outgoing sender for Message frames
    tx: UnboundedSender<Message>,
    /// A queue of incoming `ServerMessage`s (decoded from msgpack)
    incoming: UnboundedReceiver<ServerMessage>,
    /// A task handle for the background reading loop
    read_task: tokio::task::JoinHandle<()>,
}

impl Client {
    /// Connect to the given WebSocket URL and return a new `Client`.
    pub async fn connect(server_uri: &str) -> Result<Client> {
        let (ws_stream, _response) = connect_async(server_uri).await?;
        println!("[Client] Connected to {server_uri}");

        // Split into write and read halves
        let (mut ws_write, mut ws_read) = ws_stream.split();

        // We'll create an unbounded channel for sending messages to the ws_write side:
        let (tx, mut rx): (
            mpsc::UnboundedSender<Message>,
            mpsc::UnboundedReceiver<Message>,
        ) = mpsc::unbounded_channel();

        // Also create an unbounded channel for the incoming server messages:
        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();

        // Spawn a writer task that takes messages from `rx` and sends them to the server:
        // The writer side does not decode or encode anything at this point; we pass it already-packed data.
        let writer_task = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let Err(e) = ws_write.send(msg).await {
                    eprintln!("[Client] WS write error: {:?}", e);
                    break;
                }
            }
        });

        // Spawn a reader task that reads from ws_read, attempts to decode msgpack, and forwards to incoming_tx
        let reader_task = tokio::spawn(async move {
            while let Some(Ok(msg)) = ws_read.next().await {
                match msg {
                    Message::Binary(bin) => {
                        // Decode via rmp-serde
                        match decode::from_slice::<ServerMessage>(&bin) {
                            Ok(server_msg) => {
                                if incoming_tx.send(server_msg).is_err() {
                                    eprintln!("[Client] Failed to queue incoming message");
                                }
                            }
                            Err(e) => {
                                eprintln!("[Client] Failed to decode msgpack: {:?}", e);
                            }
                        }
                    }
                    Message::Text(txt) => {
                        eprintln!("[Client] Unexpected text message from server: {:?}", txt);
                        // You could also generate an "error" message here
                    }
                    Message::Close(_) => {
                        println!("[Client] Server closed the connection");
                        break;
                    }
                    Message::Ping(_) | Message::Pong(_) => {
                        // ignore pings/pongs, or handle as needed
                    }
                    _ => {
                        eprintln!("[Client] Unexpected message type from server");
                    }
                }
            }
            println!("[Client] Reader task ended.");
        });

        // We'll join the writer_task on drop or when close() is called, but let's keep only the
        // reader task handle. We can embed the writer handle in the client or not.
        let read_task = reader_task; // keep handle so we can wait/cancel if needed.

        Ok(Client {
            tx,
            incoming: incoming_rx,
            read_task,
        })
    }

    /// Close the connection. This signals the writer channel to terminate
    /// and also awaits the read task finishing.
    pub async fn close(self) -> Result<()> {
        // Attempt to send a Close message
        let _ = self.tx.send(Message::Close(None));
        // The writer side might flush out. Let's drop our sender so the writer can exit:
        drop(self.tx);

        // Wait for the read_task to complete
        self.read_task.abort();
        let _ = self.read_task.await;

        println!("[Client] Connection closed.");
        Ok(())
    }

    /// Helper: sends a serialized msgpack message to the server.
    fn send_msg(&self, msg: &ClientMessage) -> Result<()> {
        let encoded = encode::to_vec_named(msg)?; // rmp-serde encoding
        self.tx.send(Message::Binary(encoded.into()))?;
        Ok(())
    }

    /// Wait for the *next* incoming server message (FIFO).
    pub async fn wait_for_next_message(&mut self) -> Option<ServerMessage> {
        self.incoming.recv().await
    }

    // ---- High-level actions, akin to the Python client code: ---- //

    pub async fn query_existence(&mut self, program_hash: &str) -> Result<ServerMessage> {
        let msg = ClientMessage::QueryExistence {
            hash: program_hash.to_string(),
        };
        self.send_msg(&msg)?;
        // Wait for next incoming message
        while let Some(response) = self.wait_for_next_message().await {
            match response {
                ServerMessage::QueryResponse { hash, exists } => {
                    if hash == program_hash {
                        return Ok(ServerMessage::QueryResponse { hash, exists });
                    } else {
                        // If mismatch, keep waiting or handle differently
                        eprintln!("Got query response for the wrong hash: {}", hash);
                    }
                }
                ServerMessage::Error { error } => {
                    // Possibly the server returned an error?
                    return Ok(ServerMessage::Error { error });
                }
                other => {
                    // If it's not the query response, we can either keep waiting or
                    // push it back somewhere. For a simple approach, let's just keep waiting.
                    eprintln!("[Client] Unexpected message: {:?}", other);
                }
            }
        }
        anyhow::bail!("No response received from server for query_existence")
    }

    /// Upload a WASM file in chunked form.  
    /// Prints out the server ack messages as they arrive.
    pub async fn upload_program(&mut self, wasm_bytes: &[u8], program_hash: &str) -> Result<()> {
        let total_size = wasm_bytes.len();
        let total_chunks = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        let mut chunk_index = 0;
        while chunk_index < total_chunks {
            let start = chunk_index * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(total_size);
            let chunk_data = &wasm_bytes[start..end];

            let msg = ClientMessage::UploadProgram {
                hash: program_hash.to_string(),
                chunk_index,
                total_chunks,
                chunk_data: Vec::from(chunk_data),
            };
            self.send_msg(&msg)?;

            // For each chunk, we expect an upload_ack. Possibly also an upload_complete.
            // We'll loop reading from the queue until we see the ack for our chunk.
            loop {
                if let Some(incoming) = self.wait_for_next_message().await {
                    match incoming {
                        ServerMessage::UploadAck {
                            hash,
                            chunk_index: ack_idx,
                        } => {
                            if hash == program_hash && ack_idx == chunk_index {
                                // println!(
                                //     "[Client] Received ack for chunk {}/{}",
                                //     ack_idx, total_chunks
                                // );
                                break; // proceed to next chunk
                            } else {
                                eprintln!(
                                    "UploadAck mismatch: got hash={}, idx={} but expected {} and {}",
                                    hash, ack_idx, program_hash, chunk_index
                                );
                                // keep waiting or handle as error
                            }
                        }
                        ServerMessage::Error { error } => {
                            anyhow::bail!("Server returned error during upload: {}", error);
                        }
                        other => {
                            if let ServerMessage::UploadComplete {
                                hash: completed_hash,
                            } = &other
                            {
                                // Possibly the server also sends UploadComplete right after last chunk
                                if completed_hash == &program_hash {
                                    println!(
                                        "[Client] Received upload_complete for {}",
                                        completed_hash
                                    );
                                    // It's possible the server didn't wait for the chunk ack?
                                    // We'll handle that gracefully: if chunk_index is last, we are done.
                                } else {
                                    eprintln!(
                                        "UploadComplete mismatch for hash: {}",
                                        completed_hash
                                    );
                                }
                            }
                            // keep waiting for an ack that matches chunk_index
                            eprintln!(
                                "[Client] Unexpected message while waiting for chunk ack: {:?}",
                                other
                            );
                        }
                    }
                } else {
                    anyhow::bail!("Upload: No more messages from server?");
                }
            }
            chunk_index += 1;
        }

        // Now, after we've sent all chunks, we also want the `upload_complete`.
        // The server might have sent it already as part of the loop above, or it might come after the final ack.
        // We'll do a short loop to look for it, or you can do a timed wait, etc.
        loop {
            if let Ok(incoming) = self.incoming.try_recv() {
                match incoming {
                    ServerMessage::UploadComplete { hash } => {
                        if hash == program_hash {
                            println!("[Client] Upload complete for hash={}", hash);
                            break;
                        } else {
                            eprintln!("UploadComplete mismatch: got hash={}", hash);
                            // keep searching?
                        }
                    }
                    other => {
                        eprintln!(
                            "[Client] Received extra message after final chunk: {:?}",
                            other
                        );
                        // Could ignore or break as needed
                    }
                }
            } else {
                // No more messages to check; likely done.
                // This means the server might have sent "upload_complete" earlier or the server
                // doesn't send it at all if the program is already in disk.
                // We'll just break out.
                break;
            }
        }

        Ok(())
    }

    pub async fn start_program(&mut self, program_hash: &str) -> Result<Option<String>> {
        let msg = ClientMessage::StartProgram {
            hash: program_hash.to_string(),
        };
        self.send_msg(&msg)?;

        while let Some(incoming) = self.wait_for_next_message().await {
            match incoming {
                ServerMessage::ProgramLaunched { hash, instance_id } => {
                    if hash == program_hash {
                        return Ok(Some(instance_id));
                    } else {
                        eprintln!(
                            "start_program: got ProgramLaunched but with mismatched hash={}",
                            hash
                        );
                    }
                }
                ServerMessage::Error { error } => {
                    anyhow::bail!("Server error on start_program: {}", error);
                }
                other => {
                    // Possibly a program_event or something else, let's keep waiting
                    eprintln!(
                        "[Client] Unexpected message while waiting for ProgramLaunched: {:?}",
                        other
                    );
                }
            }
        }
        Ok(None)
    }

    pub fn send_event(&self, instance_id: &str, event_data: String) -> Result<()> {
        let msg = ClientMessage::SendEvent {
            instance_id: instance_id.to_string(),
            event_data,
        };
        self.send_msg(&msg)
    }

    pub async fn terminate_program(&mut self, instance_id: &str) -> Result<()> {
        let msg = ClientMessage::TerminateProgram {
            instance_id: instance_id.to_string(),
        };
        self.send_msg(&msg)?;

        while let Some(incoming) = self.wait_for_next_message().await {
            match incoming {
                ServerMessage::ProgramTerminated {
                    instance_id: term_id,
                    reason,
                } => {
                    if term_id == instance_id {
                        println!(
                            "[Client] ProgramTerminated for instance_id={}, because {}",
                            instance_id, reason
                        );
                        return Ok(());
                    } else {
                        eprintln!(
                            "Terminate mismatch: got instance_id={}, expecting={}",
                            term_id, instance_id
                        );
                    }
                }
                ServerMessage::Error { error } => {
                    anyhow::bail!("Server error on terminate_program: {}", error);
                }
                other => {
                    eprintln!(
                        "[Client] Unexpected message while waiting for ProgramTerminated: {:?}",
                        other
                    );
                }
            }
        }
        Ok(())
    }
}
