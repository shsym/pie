use crate::bindings;
use crate::instance::InstanceState;
use crate::messaging::Command;
use std::mem;
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::Resource;
use wasmtime_wasi::{DynPollable, IoView, Pollable, subscribe, async_trait};
//

#[derive(Debug)]
pub struct Subscription {
    id: usize,
    topic: String,
    receiver: mpsc::Receiver<String>,
    result: Option<String>,
    done: bool,
}

#[async_trait]
impl Pollable for Subscription {
    async fn ready(&mut self) {
        // if results are already computed, return
        if self.done {
            return;
        }

        if let Some(result) = self.receiver.recv().await {
            self.result = Some(result);
        } else {
            self.done = true;
        }
    }
}

impl bindings::wit::symphony::app::messaging::Host for InstanceState {
    async fn send(&mut self, message: String) -> Result<(), wasmtime::Error> {
        self.broadcast(self.id().to_string(), message).await?;
        Ok(())
    }

    async fn receive(&mut self) -> Result<Resource<Subscription>, wasmtime::Error> {
        self.subscribe(self.id().to_string()).await
    }

    async fn broadcast(&mut self, topic: String, message: String) -> Result<(), wasmtime::Error> {
        Command::Broadcast { topic, message }.dispatch()?;
        Ok(())
    }

    async fn subscribe(
        &mut self,
        topic: String,
    ) -> Result<Resource<Subscription>, wasmtime::Error> {
        let (tx, rx) = mpsc::channel(64);
        let (sub_tx, sub_rx) = oneshot::channel();

        Command::Subscribe {
            topic: topic.clone(),
            sender: tx,
            sub_id: sub_tx,
        }
        .dispatch()?;

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

impl bindings::wit::symphony::app::messaging::HostSubscription for InstanceState {
    async fn poll(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<Resource<DynPollable>, wasmtime::Error> {
        subscribe(self.table(), this)
    }

    async fn receive(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<Option<String>, wasmtime::Error> {
        let mut sub = self.table().get_mut(&this)?;
        Ok(mem::take(&mut sub.result))
    }

    async fn unsubscribe(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<(), wasmtime::Error> {
        let mut sub = self.table().get_mut(&this)?;
        sub.done = true;
        let topic = sub.topic.clone();
        let sub_id = sub.id;
        Command::Unsubscribe { topic, sub_id }.dispatch()?;

        Ok(())
    }

    async fn drop(&mut self, this: Resource<Subscription>) -> anyhow::Result<(), wasmtime::Error> {
        let _ = self.table().delete(this)?;
        Ok(())
    }
}
