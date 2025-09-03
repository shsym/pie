use anyhow::Result;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use rmp_serde::{decode, encode};
use std::sync::Arc;

use crate::instance::InstanceId;
use crate::server::{CHUNK_SIZE_BYTES, ClientMessage, QUERY_PROGRAM_EXISTS, ServerMessage};
use crate::utils::IdPool;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::{mpsc, oneshot};
use tokio::task;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use uuid::Uuid;

type CorrId = u32;

/// A client that interacts with the "Symphony" server.
pub struct Client {
    /// Outgoing sender for Message frames
    ws_writer_tx: UnboundedSender<Message>,
    corr_id_pool: IdPool<CorrId>,
    /// A queue of incoming `ServerMessage`s (decoded from msgpack)
    // event table
    pending_requests: Arc<DashMap<CorrId, oneshot::Sender<(bool, String)>>>,
    inst_event_tx: Arc<DashMap<InstanceId, mpsc::Sender<(String, String)>>>,
    server_event_rx: mpsc::Receiver<String>,

    /// A task handle for the background reading loop
    reader_handle: task::JoinHandle<()>,
    writer_handle: task::JoinHandle<()>,
}

#[derive(Debug)]
pub struct Instance {
    id: InstanceId,
    tx: UnboundedSender<Message>,
    event_rx: mpsc::Receiver<(String, String)>,
}

pub fn hash_program(blob: &[u8]) -> String {
    blake3::hash(blob).to_hex().to_string()
}

impl Instance {
    pub fn id(&self) -> InstanceId {
        self.id
    }

    pub async fn send<T>(&self, message: T) -> Result<()>
    where
        T: ToString,
    {
        let msg = ClientMessage::SignalInstance {
            instance_id: self.id.to_string(),
            message: message.to_string(),
        };
        self.tx
            .send(Message::Binary(encode::to_vec_named(&msg)?.into()))?;
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<(String, String)> {
        self.event_rx
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("Event channel closed"))
    }

    pub async fn terminate(&self) -> Result<()> {
        let msg = ClientMessage::TerminateInstance {
            instance_id: self.id.to_string(),
        };
        self.tx
            .send(Message::Binary(encode::to_vec_named(&msg)?.into()))?;
        Ok(())
    }
}

impl Client {
    pub async fn connect(ws_host: &str) -> Result<Client> {
        let (ws_stream, _response) = connect_async(ws_host).await?;
        //println!("[Client] Connected to {ws_host}");

        let (mut ws_write, mut ws_read) = ws_stream.split();

        let (ws_writer_tx, mut ws_writer_rx) = unbounded_channel();

        let pending_requests: Arc<DashMap<CorrId, oneshot::Sender<(bool, String)>>> =
            Arc::new(DashMap::new());
        let inst_event_tx: Arc<DashMap<InstanceId, mpsc::Sender<(String, String)>>> =
            Arc::new(DashMap::new());
        let (server_event_tx, server_event_rx) = mpsc::channel(64);

        let writer_handle = task::spawn(async move {
            while let Some(msg) = ws_writer_rx.recv().await {
                if let Err(e) = ws_write.send(msg).await {
                    eprintln!("[Client] WS write error: {:?}", e);
                    break;
                }
            }
        });

        let pending_requests_ = Arc::clone(&pending_requests);
        let inst_event_tx_ = Arc::clone(&inst_event_tx);
        let reader_handle = task::spawn(async move {
            while let Some(Ok(msg)) = ws_read.next().await {
                let maybe_server_msg = match msg {
                    Message::Binary(bin) => {
                        // Decode via rmp-serde
                        match decode::from_slice::<ServerMessage>(&bin) {
                            Ok(server_msg) => Some(server_msg),
                            Err(e) => {
                                eprintln!("[Client] Failed to decode msgpack: {:?}", e);
                                None
                            }
                        }
                    }
                    Message::Close(_) => {
                        //println!("[Client] Server closed the connection");
                        break;
                    }
                    _ => {
                        // ignore pings/pongs, or handle as needed
                        None
                    }
                };

                if maybe_server_msg.is_none() {
                    continue;
                }

                match maybe_server_msg.unwrap() {
                    ServerMessage::Response {
                        corr_id,
                        successful,
                        result,
                    } => {
                        // let the event loop handle this
                        if let Some((_, sender)) = pending_requests_.remove(&corr_id) {
                            let _ = sender.send((successful, result));
                        }
                    }
                    ServerMessage::InstanceEvent {
                        instance_id,
                        event,
                        message,
                    } => {
                        let inst_id = Uuid::parse_str(&instance_id).unwrap();
                        if let Some(sender) = inst_event_tx_.get(&inst_id) {
                            let _ = sender
                                .send(((event as u32).to_string(), message))
                                .await
                                .ok();
                        }
                    }
                    ServerMessage::ServerEvent { message } => {
                        server_event_tx.send(message).await.unwrap();
                    }
                }
            }
        });

        // We'll join the writer_task on drop or when close() is called, but let's keep only the
        // reader task handle. We can embed the writer handle in the client or not.

        Ok(Client {
            ws_writer_tx,
            corr_id_pool: IdPool::new(CorrId::MAX),
            pending_requests,
            inst_event_tx,
            server_event_rx,
            reader_handle,
            writer_handle,
        })
    }

    /// Close the connection. This signals the writer channel to terminate
    /// and also awaits the read task finishing.
    pub async fn close(self) -> Result<()> {
        // Attempt to send a Close message
        let _ = self.ws_writer_tx.send(Message::Close(None));
        // The writer side might flush out. Let's drop our sender so the writer can exit:
        drop(self.ws_writer_tx);

        // Wait for the read_task to complete
        self.reader_handle.abort();
        let _ = self.reader_handle.await;

        Ok(())
    }

    /// Helper: sends a serialized msgpack message to the server.
    fn send_msg(&self, msg: ClientMessage) -> Result<()> {
        let encoded = encode::to_vec_named(&msg)?; // rmp-serde encoding
        self.ws_writer_tx.send(Message::Binary(encoded.into()))?;
        Ok(())
    }

    async fn send_msg_and_wait(&mut self, mut msg: ClientMessage) -> Result<(bool, String)> {
        let corr_id_new = self.corr_id_pool.acquire()?;

        match &mut msg {
            ClientMessage::Authenticate { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::UploadProgram { corr_id, .. }
            | ClientMessage::LaunchInstance { corr_id, .. }
            | ClientMessage::LaunchServerInstance { corr_id, .. } => *corr_id = corr_id_new,
            _ => {
                anyhow::bail!("Invalid message type for sending and waiting");
            }
        };

        let (tx, rx) = oneshot::channel();
        self.pending_requests.insert(corr_id_new, tx);

        self.send_msg(msg)?;

        let (successful, result) = rx.await?;

        // release the corr_id
        self.corr_id_pool.release(corr_id_new)?;

        Ok((successful, result))
    }

    pub async fn authenticate(&mut self, token: &str) -> Result<()> {
        let msg = ClientMessage::Authenticate {
            corr_id: 0,
            token: token.to_string(),
        };
        let (successful, result) = self.send_msg_and_wait(msg).await?;

        if successful {
            Ok(())
        } else {
            anyhow::bail!("Authentication failed: {}", result);
        }
    }

    pub async fn query<T>(&mut self, subject: T, record: String) -> Result<String>
    where
        T: ToString,
    {
        let msg = ClientMessage::Query {
            corr_id: 0,
            subject: subject.to_string(),
            record,
        };

        let (successful, result) = self.send_msg_and_wait(msg).await?;

        if successful {
            Ok(result)
        } else {
            anyhow::bail!("Query failed: {}", result);
        }
    }

    pub async fn program_exists(&mut self, program_hash: &str) -> Result<bool> {
        self.query(QUERY_PROGRAM_EXISTS, program_hash.to_string())
            .await
            .map(|r| r == "true")
    }

    /// Upload a WASM file in chunked form.
    /// Prints out the server ack messages as they arrive.
    pub async fn upload_program(&mut self, blob: &[u8]) -> Result<()> {
        let program_hash = hash_program(blob);

        let (tx, rx) = oneshot::channel();
        let corr_id = self.corr_id_pool.acquire()?;

        self.pending_requests.insert(corr_id, tx);

        let total_size = blob.len();
        let total_chunks = total_size.div_ceil(CHUNK_SIZE_BYTES);

        let mut chunk_index = 0;
        while chunk_index < total_chunks {
            let start = chunk_index * CHUNK_SIZE_BYTES;
            let end = (start + CHUNK_SIZE_BYTES).min(total_size);
            let chunk_data = &blob[start..end];

            let msg = ClientMessage::UploadProgram {
                corr_id,
                program_hash: program_hash.to_string(),
                chunk_index,
                total_chunks,
                chunk_data: Vec::from(chunk_data),
            };
            self.send_msg(msg)?;

            chunk_index += 1;
        }

        let (successful, result) = rx.await?;

        // release the corr_id
        self.corr_id_pool.release(corr_id)?;
        if successful {
            Ok(())
        } else {
            anyhow::bail!("Query failed: {}", result);
        }
    }

    pub async fn launch_instance(
        &mut self,
        program_hash: &str,
        arguments: Vec<String>,
    ) -> Result<Instance> {
        let msg = ClientMessage::LaunchInstance {
            corr_id: 0,
            program_hash: program_hash.to_string(),
            arguments,
        };

        let (successful, result) = self.send_msg_and_wait(msg).await?;

        if successful {
            let inst_id = Uuid::parse_str(&result)?;

            let (tx, rx) = mpsc::channel(64);
            let instance = Instance {
                id: inst_id,
                tx: self.ws_writer_tx.clone(),
                event_rx: rx,
            };

            self.inst_event_tx.insert(inst_id, tx);

            Ok(instance)
        } else {
            anyhow::bail!("Query failed: {}", result);
        }
    }

    pub async fn launch_server_instance(
        &mut self,
        program_hash: &str,
        port: u32,
        arguments: Vec<String>,
    ) -> Result<()> {
        let msg = ClientMessage::LaunchServerInstance {
            corr_id: 0,
            port,
            program_hash: program_hash.to_string(),
            arguments,
        };

        let (successful, result) = self.send_msg_and_wait(msg).await?;

        if successful {
            Ok(())
        } else {
            anyhow::bail!("Query failed: {}", result);
        }
    }
}
