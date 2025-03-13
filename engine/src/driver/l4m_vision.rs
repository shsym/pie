use crate::driver::{
    Batchable, Batcher, BatchingStrategy, DriverError, KorTStrategy, StreamId, l4m,
};
use crate::instance_old::Id as InstanceId;
use crate::{backend, utils};

pub const PROTOCOL: &str = "l4m-vision"; // for future backward compatibility
mod pb_bindings {
    include!(concat!(env!("OUT_DIR"), "/l4m.vision.rs"));
}

pub trait ExecuteCommand:
    backend::ExecuteCommand<pb_bindings::Request, pb_bindings::Response>
{
}

impl<T> ExecuteCommand for T where
    T: backend::ExecuteCommand<pb_bindings::Request, pb_bindings::Response>
{
}

#[derive(Debug)]
pub enum Command {
    EmbedAudio { message: String },
    EmbedImage { message: String },
    EmbedVideo { message: String },
}

impl Batchable<BatchGroup> for Command {
    fn strategy(&self) -> Box<dyn BatchingStrategy> {
        match self {
            Command::EmbedAudio { .. } => KorTStrategy::eager().into_box(),
            Command::EmbedImage { .. } => KorTStrategy::eager().into_box(),
            Command::EmbedVideo { .. } => KorTStrategy::eager().into_box(),
        }
    }

    fn group(&self) -> BatchGroup {
        match self {
            Command::EmbedAudio { .. } => BatchGroup::EmbedAudio,
            Command::EmbedImage { .. } => BatchGroup::EmbedImage,
            Command::EmbedVideo { .. } => BatchGroup::EmbedVideo,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum BatchGroup {
    EmbedAudio,
    EmbedImage,
    EmbedVideo,
}

#[derive(Debug)]
pub struct Driver<B> {
    backend: B,
    cmd_id_pool: utils::IdPool<u32>,
    objects: l4m::ObjectView,
    cmd_batcher: Batcher<Command, (InstanceId, StreamId), BatchGroup>,
}

impl<B> Driver<B>
where
    B: ExecuteCommand,
{
    pub async fn new(backend: B, objects: l4m::ObjectView) -> Self {
        Self {
            backend,
            cmd_id_pool: utils::IdPool::new(u32::MAX),
            objects,
            cmd_batcher: Batcher::new(),
        }
    }

    pub fn submit(
        &mut self,
        inst: InstanceId,
        stream: StreamId,
        cmd: Command,
    ) -> Result<(), DriverError> {
        match cmd {
            Command::EmbedImage { message } => {
                let correlation_id = self
                    .cmd_id_pool
                    .acquire()
                    .map_err(|e| DriverError::LockError)?;

                let msg = pb_bindings::Request {
                    correlation_id,
                    command: Some(pb_bindings::request::Command::EmbedImage(
                        pb_bindings::BatchEmbedImage { items: vec![] },
                    )),
                };

                // self.backend
                //     .exec(msg)
                //     .await
                //     .map_err(|_| DriverError::SendError)?;

                Ok(())
            }
        }
    }
}

#[derive(Clone)]
pub struct Simulator {}

impl backend::Simulate<pb_bindings::Request, pb_bindings::Response> for Simulator {
    fn simulate(&mut self, cmd: pb_bindings::Request) -> Option<pb_bindings::Response> {
        todo!()
    }
}
