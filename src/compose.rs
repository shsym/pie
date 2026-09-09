use std::net::{Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use controller_api::{Ack, GatewayInfo, Neighbors, RoutingTable, WorkerInfo, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use worker::ControlLink;

#[derive(Clone)]
struct EmbeddedControl(controller::Handle);

impl ControlLink for EmbeddedControl {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        Ok(self.0.register_worker(info).await)
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        Ok(self.0.heartbeat(id).await)
    }

    async fn report_worker(&self, id: WorkerId, status: WorkerStatus) -> Result<()> {
        self.0.report_worker(id, status).await;
        Ok(())
    }

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        self.0.worker_watch(id)
    }
}

impl gateway::GatewayControl for EmbeddedControl {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        Ok(self.0.register_gateway(info).await)
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        Ok(self.0.heartbeat(id).await)
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.0.gateway_watch()
    }
}

pub struct StandaloneHandle {
    pub listen_addr: SocketAddr,
    pub worker_addr: SocketAddr,
    _controller: controller::Handle,
    worker: worker::WorkerHandle,
    gateway: JoinHandle<()>,
}

impl StandaloneHandle {
    pub async fn shutdown(self) {
        self.gateway.abort();
        self.worker.shutdown().await;
    }
}

pub async fn run_standalone(
    controller: controller::Config,
    mut gateway: gateway::Config,
    worker: worker::Config,
) -> Result<StandaloneHandle> {
    bootstrap::install_crypto_provider();

    let handle = controller::embed(controller);
    let control = EmbeddedControl(handle.clone());

    gateway.worker_listen = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));

    let host: std::net::IpAddr = worker.server.host.parse().with_context(|| {
        format!(
            "[server] host {:?} is not an IP address",
            worker.server.host
        )
    })?;
    gateway.listen = SocketAddr::new(host, worker.server.port);
    let gw = gateway::bind(gateway, control.clone())
        .await
        .context("bind in-proc gateway")?;
    let listen_addr = gw.listen_addr;
    let worker_addr = gw.worker_addr;

    let worker = worker::run_with(
        worker,
        control,
        vec![format!("tcp://{worker_addr}")],
        Some(format!("ws://{listen_addr}")),
    )
    .await
    .context("boot embedded worker")?;

    let gateway = tokio::spawn(async move {
        if let Err(e) = gw.serve().await {
            tracing::error!(error = %e, "in-proc gateway exited");
        }
    });

    Ok(StandaloneHandle {
        listen_addr,
        worker_addr,
        _controller: handle,
        worker,
        gateway,
    })
}
