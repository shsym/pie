use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use client::message::ClientMessage;
use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex as TokioMutex, watch};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;

pub struct ClientServerHandle {
    pub bound: String,
    pub task: tokio::task::JoinHandle<()>,
}

pub async fn spawn(listen: &str) -> Result<ClientServerHandle> {
    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind client server on {listen}"))?;
    let bound = format!(
        "ws://{}",
        listener.local_addr().context("client server local_addr")?
    );

    let task = tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    let _ = stream.set_nodelay(true);
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream).await {
                            tracing::warn!(?peer, error = %e, "client connection ended");
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "client server accept failed; stopping");
                    break;
                }
            }
        }
    });

    Ok(ClientServerHandle { bound, task })
}

async fn handle_connection(stream: TcpStream) -> Result<()> {
    let ws = accept_async(stream).await.context("websocket handshake")?;
    let (tx, mut rx) = ws.split();

    let client_id = runtime::server::open_session().map_err(|e| anyhow!("open session: {e}"))?;

    let ws_tx = Arc::new(TokioMutex::new(tx));
    let (stop_tx, stop_rx) = watch::channel(false);

    let poll_task = {
        let ws_tx = Arc::clone(&ws_tx);
        let mut stop_rx = stop_rx;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    changed = stop_rx.changed() => {
                        if changed.is_err() || *stop_rx.borrow() {
                            break;
                        }
                    }
                    poll = runtime::server::recv_messages(client_id, 200, 64) => {
                        let messages = match poll {
                            Ok(m) => m,
                            Err(e) => {
                                tracing::warn!(session = client_id, error = %e, "session recv failed");
                                break;
                            }
                        };
                        for message in messages {
                            let encoded = match rmp_serde::to_vec_named(&message) {
                                Ok(bytes) => bytes,
                                Err(e) => {
                                    tracing::warn!(session = client_id, error = %e, "encode ServerMessage failed");
                                    continue;
                                }
                            };
                            let mut guard = ws_tx.lock().await;
                            if guard.send(WsMessage::Binary(encoded.into())).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            }
        })
    };

    while let Some(frame) = rx.next().await {
        let bytes = match frame.context("websocket read")? {
            WsMessage::Binary(b) => b,
            WsMessage::Close(_) => break,
            _ => continue,
        };
        let msg: ClientMessage =
            rmp_serde::from_slice(bytes.as_ref()).context("decode ClientMessage")?;
        runtime::server::send_client_message(client_id, msg)
            .map_err(|e| anyhow!("send client message: {e}"))?;
    }

    let _ = stop_tx.send(true);
    let _ = poll_task.await;
    runtime::server::close_session(client_id);
    Ok(())
}
