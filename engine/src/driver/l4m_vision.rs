use crate::driver::{l4m, DriverError, StreamId};
use crate::instance::Id as InstanceId;
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

pub enum Command {
    EmbedImage { message: String },
}

#[derive(Debug)]
pub struct Driver<B> {
    backend: B,
    cmd_id_pool: utils::IdPool<u32>,
    objects: l4m::ObjectRegistryView,
    //cmd_batcher: CommandBatcher,
}

impl<B> Driver<B>
where
    B: ExecuteCommand,
{
    pub async fn new(backend: B, objects: l4m::ObjectRegistryView) -> Self {
        Self {
            backend,
            cmd_id_pool: utils::IdPool::new(u32::MAX),
            objects
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
