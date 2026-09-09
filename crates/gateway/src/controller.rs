use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, GatewayInfo, RoutingTable};
use ids::{GatewayId, NodeId};
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

const WATCH_DEADLINE: Duration = Duration::from_secs(300);

const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

pub trait GatewayControl: Clone + Send + Sync + 'static {
    fn register_gateway(&self, info: GatewayInfo)
    -> impl Future<Output = Result<GatewayId>> + Send;

    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    fn routing_watch(&self) -> watch::Receiver<RoutingTable>;
}

impl GatewayControl for ControlClient {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        self.register_gateway(tarpc::context::current(), info)
            .await
            .context("register_gateway rpc")
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        self.heartbeat(tarpc::context::current(), id)
            .await
            .context("heartbeat rpc")
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        let (tx, rx) = watch::channel(RoutingTable {
            epoch: 0,
            workers: Vec::new(),
        });
        tokio::spawn(watch_routing_loop(self.clone(), tx));
        rx
    }
}

pub(crate) async fn heartbeat_loop<C: GatewayControl>(
    control: C,
    mut id: GatewayId,
    info: GatewayInfo,
) {
    let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
    loop {
        ticker.tick().await;
        match control.heartbeat(NodeId::Gateway(id)).await {
            Ok(Ack::Ok) => {}
            Ok(Ack::ReRegister) => {
                tracing::warn!(%id, "controller lost our registration; re-registering");
                match control.register_gateway(info.clone()).await {
                    Ok(new_id) => {
                        id = new_id;
                        tracing::info!(%id, "gateway re-registered");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "re-register failed; retrying next tick");
                    }
                }
            }
            Err(e) => tracing::warn!(error = %e, "heartbeat transport error"),
        }
    }
}

async fn watch_routing_loop(control: ControlClient, routing_tx: watch::Sender<RoutingTable>) {
    let mut since: u64 = 0;
    loop {
        let mut ctx = tarpc::context::current();
        ctx.deadline = Instant::now() + WATCH_DEADLINE;
        match control.watch_gateway(ctx, since).await {
            Ok(table) => {
                since = table.epoch;
                tracing::debug!(
                    epoch = since,
                    workers = table.workers.len(),
                    "routing table updated"
                );
                if routing_tx.send(table).is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "watch_gateway error; retrying");
                tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
            }
        }
    }
}

pub(crate) async fn connect_controller(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        #[cfg(unix)]
        {
            let conn = unix::connect(path, Bincode::default)
                .await
                .with_context(|| format!("dialing controller at {addr}"))?;
            Ok(ControlClient::new(cfg, conn).spawn())
        }
        #[cfg(not(unix))]
        {
            let _ = (path, cfg);
            anyhow::bail!(
                "{addr}: a `unix://` control address is distributed serving, which needs a unix-domain socket; this build is single-node and speaks `tcp://`"
            )
        }
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        let _ = conn.get_ref().set_nodelay(true);
        Ok(ControlClient::new(cfg, conn).spawn())
    }
}
