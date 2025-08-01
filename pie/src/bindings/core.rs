use crate::instance::InstanceState;
use crate::messaging::{PubSubCommand, PushPullCommand, dispatch_i2i, dispatch_u2i};
use crate::model::{self, Command, StreamPriority};
use crate::{bindings, kvs, server};
use std::mem;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};

// A counter to generate unique stream IDs for new queues
static NEXT_STREAM_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub service_id: usize,
}

#[derive(Debug, Clone)]
pub struct Queue {
    pub service_id: usize,
    pub stream_id: u32,
}

#[derive(Debug)]
pub struct DebugQueryResult {
    receiver: oneshot::Receiver<String>,
    result: Option<String>,
    done: bool,
}

#[derive(Debug)]
pub struct SynchronizationResult {
    receiver: oneshot::Receiver<()>,
    done: bool,
}

#[derive(Debug)]
pub struct ReceiveResult {
    receiver: oneshot::Receiver<String>,
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
impl Pollable for DebugQueryResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let res = (&mut self.receiver).await.unwrap();
        self.result = Some(res);
        self.done = true;
    }
}

#[async_trait]
impl Pollable for ReceiveResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let res = (&mut self.receiver).await.unwrap();
        self.result = Some(res);
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

#[async_trait]
impl Pollable for SynchronizationResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let _ = (&mut self.receiver).await.unwrap();
        self.done = true;
    }
}

impl bindings::pie::inferlet::core::Host for InstanceState {
    async fn get_version(&mut self) -> anyhow::Result<String> {
        let (tx, rx) = oneshot::channel();
        crate::runtime::Command::GetVersion { event: tx }.dispatch()?;
        rx.await.map_err(Into::into)
    }

    async fn get_instance_id(&mut self) -> anyhow::Result<String> {
        Ok(self.id().to_string())
    }

    async fn get_arguments(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(self.arguments().to_vec())
    }

    async fn debug_query(&mut self, query: String) -> anyhow::Result<Resource<DebugQueryResult>> {
        let (tx, rx) = oneshot::channel();

        let res = DebugQueryResult {
            receiver: rx,
            result: None,
            done: false,
        };

        crate::runtime::Command::DebugQuery { query, event: tx }.dispatch()?;
        Ok(self.table().push(res)?)
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
            receiver: rx,
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

    async fn get_model(&mut self, name: String) -> anyhow::Result<Option<Resource<Model>>> {
        if let Some(service_id) = model::model_service_id(&name) {
            let model = Model { name, service_id };
            let res = self.table().push(model)?;
            return Ok(Some(res));
        }
        Ok(None)
    }

    async fn get_all_models(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(model::available_models())
    }

    async fn get_all_models_with_traits(
        &mut self,
        _traits: Vec<String>,
    ) -> anyhow::Result<Vec<String>> {
        // Placeholder: Implement trait filtering logic
        Ok(model::available_models())
    }

    async fn store_set(&mut self, key: String, value: String) -> anyhow::Result<()> {
        kvs::dispatch(kvs::Command::Set { key, value })?;
        Ok(())
    }
    async fn store_get(&mut self, key: String) -> anyhow::Result<Option<String>> {
        let (tx, rx) = oneshot::channel();
        kvs::dispatch(kvs::Command::Get { key, response: tx })?;
        let res = rx.await?;
        Ok(res)
    }

    async fn store_delete(&mut self, key: String) -> anyhow::Result<()> {
        kvs::dispatch(kvs::Command::Delete { key })?;
        Ok(())
    }

    async fn store_exists(&mut self, key: String) -> anyhow::Result<bool> {
        let (tx, rx) = oneshot::channel();
        kvs::dispatch(kvs::Command::Exists { key, response: tx })?;
        let res = rx.await?;
        Ok(res)
    }
    async fn store_list_keys(&mut self) -> anyhow::Result<Vec<String>> {
        let (tx, rx) = oneshot::channel();
        kvs::dispatch(kvs::Command::ListKeys { response: tx })?;
        let res = rx.await?;
        Ok(res)
    }
}

impl bindings::pie::inferlet::core::HostModel for InstanceState {
    async fn get_name(&mut self, this: Resource<Model>) -> anyhow::Result<String> {
        Ok(self.table().get(&this)?.name.clone())
    }
    async fn get_traits(&mut self, _this: Resource<Model>) -> anyhow::Result<Vec<String>> {
        // Placeholder
        Ok(vec![
            "input_text".to_string(),
            "tokenize".to_string(),
            "output_text".to_string(),
        ])
    }
    async fn get_description(&mut self, _this: Resource<Model>) -> anyhow::Result<String> {
        // Placeholder
        Ok("A large language model.".to_string())
    }
    async fn get_version(&mut self, _this: Resource<Model>) -> anyhow::Result<String> {
        // Placeholder
        Ok("1.0".to_string())
    }
    async fn get_license(&mut self, _this: Resource<Model>) -> anyhow::Result<String> {
        // Placeholder
        Ok("Proprietary".to_string())
    }
    async fn get_prompt_template(&mut self, _this: Resource<Model>) -> anyhow::Result<String> {
        // Placeholder
        Ok("".to_string())
    }

    async fn get_service_id(&mut self, this: Resource<Model>) -> anyhow::Result<u32> {
        Ok(self.table().get(&this)?.service_id as u32)
    }

    async fn create_queue(&mut self, this: Resource<Model>) -> anyhow::Result<Resource<Queue>> {
        let model = self.table().get(&this)?;
        let queue = Queue {
            service_id: model.service_id,
            stream_id: NEXT_STREAM_ID.fetch_add(1, Ordering::SeqCst),
        };
        let res = self.table().push(queue)?;
        Ok(res)
    }

    async fn drop(&mut self, this: Resource<Model>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::core::HostQueue for InstanceState {
    async fn get_service_id(&mut self, this: Resource<Queue>) -> anyhow::Result<u32> {
        Ok(self.table().get(&this)?.service_id as u32)
    }

    async fn synchronize(
        &mut self,
        this: Resource<Queue>,
    ) -> anyhow::Result<Resource<SynchronizationResult>> {
        let inst_id = self.id();
        let queue = self.table().get(&this)?;
        let (tx, rx) = oneshot::channel();
        Command::Synchronize {
            inst_id,
            stream_id: queue.stream_id,
            handle: tx,
        }
        .dispatch(queue.service_id)?;

        let result = SynchronizationResult {
            receiver: rx,
            done: false,
        };
        Ok(self.table().push(result)?)
    }

    async fn set_priority(
        &mut self,
        this: Resource<Queue>,
        priority: bindings::pie::inferlet::core::Priority,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let queue = self.table().get(&this)?;
        Command::SetStreamPriority {
            inst_id,
            stream_id: queue.stream_id,
            priority: match priority {
                bindings::pie::inferlet::core::Priority::High => StreamPriority::High,
                bindings::pie::inferlet::core::Priority::Normal => StreamPriority::Normal,
                bindings::pie::inferlet::core::Priority::Low => StreamPriority::Low,
            },
        }
        .dispatch(queue.service_id)?;
        Ok(())
    }

    async fn debug_query(
        &mut self,
        this: Resource<Queue>,
        query: String,
    ) -> anyhow::Result<Resource<DebugQueryResult>> {
        let inst_id = self.id();
        let queue = self.table().get(&this)?;
        // Placeholder
        let (tx, rx) = oneshot::channel();

        let res = DebugQueryResult {
            receiver: rx,
            result: None,
            done: false,
        };

        Command::DebugQuery {
            inst_id,
            stream_id: queue.stream_id,
            query,
            handle: tx,
        }
        .dispatch(queue.service_id)?;
        Ok(self.table().push(res)?)
    }

    async fn drop(&mut self, this: Resource<Queue>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::core::HostDebugQueryResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<DebugQueryResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(&mut self, this: Resource<DebugQueryResult>) -> anyhow::Result<Option<String>> {
        let result = self.table().get_mut(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<DebugQueryResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::core::HostSynchronizationResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<SynchronizationResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(&mut self, this: Resource<SynchronizationResult>) -> anyhow::Result<Option<bool>> {
        let result = self.table().get_mut(&this)?;
        if result.done {
            Ok(Some(true))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<SynchronizationResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::core::HostReceiveResult for InstanceState {
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

impl bindings::pie::inferlet::core::HostSubscription for InstanceState {
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
