use crate::bindings2;
use crate::instance::InstanceState;
use crate::l4m::{self, Command, StreamPriority};
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::oneshot;
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
pub struct SynchronizationResult {
    receiver: Option<oneshot::Receiver<()>>,
    done: bool,
}

#[async_trait]
impl Pollable for SynchronizationResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(receiver) = self.receiver.take() {
            let _ = receiver.await.unwrap();
        }
        self.done = true;
    }
}

impl bindings2::pie::inferlet::model::Host for InstanceState {
    async fn get_model(&mut self, name: String) -> anyhow::Result<Option<Resource<Model>>> {
        if let Some(service_id) = l4m::model_service_id(&name) {
            let model = Model { name, service_id };
            let res = self.table().push(model)?;
            return Ok(Some(res));
        }
        Ok(None)
    }

    async fn get_all_models(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(l4m::available_models())
    }

    async fn get_all_models_with_traits(
        &mut self,
        _traits: Vec<String>,
    ) -> anyhow::Result<Vec<String>> {
        // Placeholder: Implement trait filtering logic
        Ok(l4m::available_models())
    }
}

impl bindings2::pie::inferlet::model::HostModel for InstanceState {
    async fn get_name(&mut self, this: Resource<Model>) -> anyhow::Result<String> {
        Ok(self.table().get(&this)?.name.clone())
    }
    async fn get_traits(&mut self, _this: Resource<Model>) -> anyhow::Result<Vec<String>> {
        // Placeholder
        Ok(vec![])
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
    async fn create_queue(&mut self, this: Resource<Model>) -> anyhow::Result<Resource<Queue>> {
        let model = self.table().get(&this)?;
        let queue = Queue {
            service_id: model.service_id,
            stream_id: NEXT_STREAM_ID.fetch_add(1, Ordering::SeqCst),
        };
        let res = self.table().push(queue)?;
        Ok(res)
    }
    async fn debug_query(
        &mut self,
        _this: Resource<Model>,
        _query: String,
    ) -> anyhow::Result<String> {
        // Placeholder
        Ok("Debug query not implemented.".to_string())
    }
    async fn drop(&mut self, this: Resource<Model>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings2::pie::inferlet::model::HostQueue for InstanceState {
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
            receiver: Some(rx),
            done: false,
        };
        Ok(self.table().push(result)?)
    }

    async fn set_priority(
        &mut self,
        this: Resource<Queue>,
        priority: bindings2::pie::inferlet::model::Priority,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let queue = self.table().get(&this)?;
        Command::SetStreamPriority {
            inst_id,
            stream_id: queue.stream_id,
            priority: match priority {
                bindings2::pie::inferlet::model::Priority::High => StreamPriority::High,
                bindings2::pie::inferlet::model::Priority::Normal => StreamPriority::Normal,
                bindings2::pie::inferlet::model::Priority::Low => StreamPriority::Low,
            },
        }
        .dispatch(queue.service_id)?;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Queue>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings2::pie::inferlet::model::HostSynchronizationResult for InstanceState {
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
