use crate::driver::DriverError;
use crate::lm::{ImageEmbedder, TokenEmb};
use crate::object::VspaceId;
use crate::utils::Stream;
use crate::{backend_old, driver_l4m, utils};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

pub const PROTOCOL: &str = "l4m-vision"; // for future backward compatibility

mod l4m {
    pub mod vision {
        include!(concat!(env!("OUT_DIR"), "/l4m.vision.rs"));
    }
}

//
// pub trait ExecuteCommand: backend::ExecuteCommand<l4m::Request, l4m::Response> {}
// impl<T> crate::driver_l4m::ExecuteCommand for T where T: backend::ExecuteCommand<l4m::Request, l4m::Response> {}

pub trait ExecuteCommand:
    backend_old::Protocol<l4m::vision::Request, l4m::vision::Response>
{
}

impl<T> ExecuteCommand for T where
    T: backend_old::Protocol<l4m::vision::Request, l4m::vision::Response>
{
}

#[derive(Debug)]
pub struct Driver<B> {
    backend: B,
    cmd_id_pool: utils::IdPool<u32>,
}

impl<B> Driver<B>
where
    B: ExecuteCommand,
{
    pub async fn new(b: B) -> Self {
        Self {
            backend: b,
            cmd_id_pool: utils::IdPool::new(u32::MAX),
        }
    }

    pub async fn embed_image(
        &mut self,
        message: String,
    ) -> Result<(), DriverError> {
        let correlation_id = self
            .cmd_id_pool
            .acquire()
            .map_err(|e| DriverError::LockError)?;

        // let msg = l4m::vision::Request {
        //     correlation_id,
        //     command: Some(l4m::vision::request::Command::EmbedImage(
        //         
        //     )),
        // };
        // 
        // self.event_table.insert(correlation_id, handler);
        // 
        // self.backend
        //     .exec(msg)
        //     .await
        //     .map_err(|_| DriverError::SendError)?;

        Ok(())
    }

}

#[derive(Clone)]
pub struct Simulator {}

impl backend_old::Simulate<l4m::vision::Request, l4m::vision::Response> for Simulator {
    fn simulate(&mut self, cmd: l4m::vision::Request) -> Option<l4m::vision::Response> {
        None
    }
}
