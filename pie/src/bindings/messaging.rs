use crate::instance::InstanceState;
use crate::messaging::{PubSubCommand, PushPullCommand, dispatch_i2i, dispatch_u2i};
use crate::{bindings, server};
use std::mem;
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable , subscribe};
//

#[derive(Debug)]
pub struct ReceiveResult {
    receiver: Vec<oneshot::Receiver<String>>,
    result: Vec<String>,
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
        // if results are already computed, return
        if self.done {
            return;
        }

        for rx in &mut self.receiver {
            let result = rx.await.unwrap();
            self.result.push(result);
        }
        self.done = true;
    }
}

#[async_trait]
impl Pollable for Subscription {
    async fn ready(&mut self) {
        // if results are already computed, return
        if self.done {
            return;
        }

        if let Some(result) = self.receiver.recv().await {
            //println!("Sub Received message: {:?}", result);
            self.result = Some(result);
            self.done = true;
        } else {
            self.done = true;
        }
    }
}

impl bindings::wit::pie::nbi::messaging::Host for InstanceState {
    async fn send(&mut self, message: String) -> Result<(), wasmtime::Error> {
        server::Command::Send {
            inst_id: self.id(),
            message,
        }
        .dispatch()?;
        Ok(())
    }

    async fn receive(&mut self) -> Result<Resource<ReceiveResult>, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        dispatch_u2i(PushPullCommand::Pull {
            topic: self.id().to_string(),
            message: tx,
        });

        // generate some random string

        let res = ReceiveResult {
            receiver: vec![rx],
            result: Vec::new(),
            done: false,
        };

        Ok(self.table().push(res)?)
    }

    async fn broadcast(&mut self, topic: String, message: String) -> Result<(), wasmtime::Error> {
        dispatch_i2i(PubSubCommand::Publish { topic, message });
        Ok(())
    }

    async fn subscribe(
        &mut self,
        topic: String,
    ) -> Result<Resource<Subscription>, wasmtime::Error> {
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

impl bindings::wit::pie::nbi::messaging::HostReceiveResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ReceiveResult>,
    ) -> anyhow::Result<Resource<DynPollable>, wasmtime::Error> {
        subscribe(self.table(), this)
    }

    async fn get(
        &mut self,
        this: Resource<ReceiveResult>,
    ) -> anyhow::Result<Option<String>, wasmtime::Error> {
        let sub = self.table().get_mut(&this)?;

        Ok(sub.result.get(0).cloned())
    }

    async fn drop(&mut self, this: Resource<ReceiveResult>) -> anyhow::Result<(), wasmtime::Error> {
        let _ = self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::wit::pie::nbi::messaging::HostSubscription for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<Resource<DynPollable>, wasmtime::Error> {
        subscribe(self.table(), this)
    }

    async fn get(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<Option<String>, wasmtime::Error> {
        let sub = self.table().get_mut(&this)?;
        Ok(mem::take(&mut sub.result))
    }

    async fn unsubscribe(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<(), wasmtime::Error> {
        let sub = self.table().get_mut(&this)?;
        sub.done = true;
        let topic = sub.topic.clone();
        let sub_id = sub.id;

        dispatch_i2i(PubSubCommand::Unsubscribe { topic, sub_id });

        Ok(())
    }

    async fn drop(&mut self, this: Resource<Subscription>) -> anyhow::Result<(), wasmtime::Error> {
        let _ = self.table().delete(this)?;
        Ok(())
    }
}
