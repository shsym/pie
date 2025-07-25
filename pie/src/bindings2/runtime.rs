use crate::instance::InstanceState;
use crate::messaging::{PubSubCommand, PushPullCommand, dispatch_i2i, dispatch_u2i};
use crate::runtime::Command as RuntimeCommand;
use crate::{bindings2, server};
use std::mem;
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};

#[derive(Debug)]
pub struct ReceiveResult {
    receiver: Option<oneshot::Receiver<String>>,
    result: Option<String>,
    done: bool,
}

#[derive(Debug)]
pub struct Subscription {
    id: usize,
    topic: String,
    receiver: mpsc::Receiver<String>,
    result: Option<String>,
    done: bool,
}

#[async_trait]
impl Pollable for ReceiveResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(rx) = self.receiver.take() {
            if let Ok(result) = rx.await {
                self.result = Some(result);
            }
        }
        self.done = true;
    }
}

#[async_trait]
impl Pollable for Subscription {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(result) = self.receiver.recv().await {
            self.result = Some(result);
            self.done = true;
        } else {
            self.done = true;
        }
    }
}

impl bindings2::pie::inferlet::runtime::Host for InstanceState {
    async fn get_version(&mut self) -> anyhow::Result<String> {
        let (tx, rx) = oneshot::channel();
        RuntimeCommand::GetVersion { event: tx }.dispatch()?;
        rx.await.map_err(Into::into)
    }

    async fn get_instance_id(&mut self) -> anyhow::Result<String> {
        Ok(self.id().to_string())
    }

    async fn get_arguments(&mut self) -> anyhow::Result<Vec<String>> {
        // Placeholder: Return empty args
        Ok(vec![])
    }

    async fn debug_query(&mut self, _query: String) -> anyhow::Result<String> {
        // Placeholder
        Ok("Debug query not implemented.".to_string())
    }

    async fn send(&mut self, message: String) -> anyhow::Result<()> {
        server::Command::Send {
            inst_id: self.id(),
            message,
        }
        .dispatch()?;
        Ok(())
    }

    async fn receive(&mut self) -> anyhow::Result<Resource<ReceiveResult>> {
        let (tx, rx) = oneshot::channel();
        dispatch_u2i(PushPullCommand::Pull {
            topic: self.id().to_string(),
            message: tx,
        });
        let res = ReceiveResult {
            receiver: Some(rx),
            result: None,
            done: false,
        };
        Ok(self.table().push(res)?)
    }

    async fn broadcast(&mut self, topic: String, message: String) -> anyhow::Result<()> {
        dispatch_i2i(PubSubCommand::Publish { topic, message });
        Ok(())
    }

    async fn subscribe(&mut self, topic: String) -> anyhow::Result<Resource<Subscription>> {
        let (tx, rx) = mpsc::channel(64);
        let (sub_tx, sub_rx) = oneshot::channel();
        dispatch_i2i(PubSubCommand::Subscribe {
            topic: topic.clone(),
            sender: tx,
            sub_id: sub_tx,
        });
        let sub_id = sub_rx.await?;
        let sub = Subscription {
            id: sub_id,
            topic,
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(self.table().push(sub)?)
    }
}

impl bindings2::pie::inferlet::runtime::HostReceiveResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ReceiveResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(&mut self, this: Resource<ReceiveResult>) -> anyhow::Result<Option<String>> {
        Ok(self.table().get_mut(&this)?.result.clone())
    }

    async fn drop(&mut self, this: Resource<ReceiveResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings2::pie::inferlet::runtime::HostSubscription for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(&mut self, this: Resource<Subscription>) -> anyhow::Result<Option<String>> {
        Ok(mem::take(&mut self.table().get_mut(&this)?.result))
    }

    async fn unsubscribe(&mut self, this: Resource<Subscription>) -> anyhow::Result<()> {
        let sub = self.table().get_mut(&this)?;
        sub.done = true;
        let topic = sub.topic.clone();
        let sub_id = sub.id;
        dispatch_i2i(PubSubCommand::Unsubscribe { topic, sub_id });
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Subscription>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}
