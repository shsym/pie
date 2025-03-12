use crate::instance::{Command, Id as InstanceId};
use crate::lm::{CausalLanguageModel, CausalTransformer, ImageEmbedder, KvBlock};
use crate::object::{IdMapper, VspaceId};
use crate::runtime::Runtime;
use crate::server::ServerMessage;
use crate::utils::Stream;
use crate::{driver_l4m, driver_l4m_vision, driver_ping, instance, object, utils};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ControllerError {
    #[error("Failed to acquire a vspace id")]
    VspacePoolAcquireFailed,

    #[error("Failed to release the vspace id")]
    VspacePoolReleaseFailed,

    #[error("Vspace not found for instance {0}")]
    VspaceNotFound(instance::Id),

    #[error("Instance not found: {0}")]
    InstanceNotFound(instance::Id),

    #[error("Client not found for instance {0}")]
    ClientNotFound(instance::Id),

    #[error("Driver error: {0}")]
    DriverError(String),

    #[error("Exported block resource not found: {0}")]
    ExportedBlockNotFound(String),

    #[error("Send error: {0}")]
    SendError(String),
}

pub struct Controller<B1, B2, B3> {
    state: Arc<Runtime>,

    vspaces: HashMap<instance::Id, VspaceId>,
    vspace_id_pool: utils::IdPool<VspaceId>,

    subscriptions: HashMap<String, Vec<instance::Id>>,
    exported_blocks: HashMap<String, ExportedBlocks>,

    driver_l4m: driver_l4m::Driver<B1>,
    driver_l4m_vision: driver_l4m_vision::Driver<B2>,
    driver_ping: driver_ping::Driver<B3>,
}

impl<B1,B2, B3> Controller<B1, B2, B3>
where
    B1: driver_l4m::ExecuteCommand,
    B2: driver_l4m_vision::ExecuteCommand,
    B3: driver_ping::ExecuteCommand,
{
    pub async fn new(state: Arc<Runtime>, backend_l4m: B1, backend_l4m_vision:B2, backend_ping: B3) -> Self {
        Controller {
            state,
            vspaces: HashMap::new(),
            vspace_id_pool: utils::IdPool::new(VspaceId::MAX),
            subscriptions: HashMap::new(),
            exported_blocks: HashMap::new(),
            driver_l4m: driver_l4m::Driver::new(backend_l4m).await,
            driver_l4m_vision: driver_l4m_vision::Driver::new(backend_l4m_vision).await,
            driver_ping: driver_ping::Driver::new(backend_ping).await,
        }
    }

    pub async fn submit(&mut self) -> Result<(), ControllerError> {
        self.driver_l4m
            .submit(Instant::now())
            .await
            .map_err(|e| ControllerError::DriverError(format!("Submit failed: {}", e)))
    }

    pub async fn handle_command(
        &mut self,
        inst_id: instance::Id,
        cmd: Command,
    ) -> Result<(), ControllerError> {
        if let Err(e) = self.try_handle_command(inst_id, cmd).await {
            //eprintln!("Controller error: {:?}", e);

            let inst = self
                .state
                .running_instances
                .get(&inst_id)
                .ok_or(ControllerError::InstanceNotFound(inst_id))?;

            inst.evt_from_system.send(e.to_string()).await;
        }
        Ok(())
    }

    async fn try_handle_command(
        &mut self,
        inst_id: instance::Id,
        cmd: Command,
    ) -> Result<(), ControllerError> {
        match cmd {
            Command::CreateInstance { handle } => {
                let space = self
                    .vspace_id_pool
                    .acquire()
                    .map_err(|_| ControllerError::VspacePoolAcquireFailed)?;

                self.vspaces.insert(inst_id, space);
                self.driver_l4m.init_space(space).map_err(|e| {
                    ControllerError::DriverError(format!("init_space failed: {}", e))
                })?;

                handle.send(self.driver_l4m.utils.clone());
            }
            Command::DestroyInstance => {
                let stream = Stream::new(&inst_id, None);
                let space = self
                    .vspaces
                    .remove(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.vspace_id_pool
                    .release(space)
                    .map_err(|_| ControllerError::VspacePoolReleaseFailed)?;
                self.driver_l4m.destroy_space(stream, &space).map_err(|e| {
                    ControllerError::DriverError(format!("destroy_space failed: {}", e))
                })?;

                // Remove all subscriptions
                for subs in self.subscriptions.values_mut() {
                    subs.retain(|&x| x != inst_id);
                }

                // Remove all exported blocks
                self.exported_blocks.retain(|_, v| v.owner_id != inst_id);
            }
            Command::SendToOrigin { message } => {
                let inst = self
                    .state
                    .running_instances
                    .get(&inst_id)
                    .ok_or(ControllerError::InstanceNotFound(inst_id))?;

                let server_msg = ServerMessage::ProgramEvent {
                    instance_id: inst_id.to_string(),
                    event_data: message,
                };

                inst.to_origin.send(server_msg).await.map_err(|e| {
                    ControllerError::SendError(format!("SendToOrigin failed: {}", e))
                })?;
            }
            Command::BroadcastToPeers { topic, message } => {
                if let Some(subscribers) = self.subscriptions.get(&topic) {
                    for sub in subscribers {
                        let inst = self
                            .state
                            .running_instances
                            .get(sub)
                            .ok_or(ControllerError::InstanceNotFound(*sub))?;
                        inst.evt_from_peers
                            .send((topic.clone(), message.clone()))
                            .await
                            .map_err(|e| {
                                ControllerError::SendError(format!(
                                    "BroadcastToPeers failed: {}",
                                    e
                                ))
                            })?;
                    }
                }
            }
            Command::Subscribe { topic } => {
                let subs = self.subscriptions.entry(topic).or_insert_with(Vec::new);
                subs.push(inst_id);
            }
            Command::Unsubscribe { topic } => {
                if let Some(subs) = self.subscriptions.get_mut(&topic) {
                    subs.retain(|&x| x != inst_id);
                }
            }

            Command::Ping { message, handle } => {
                self.driver_ping
                    .ping(message, handle)
                    .await
                    .map_err(|e| ControllerError::DriverError(format!("Ping failed: {}", e)))?;
            }
            Command::AllocateBlocks { stream, blocks } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .alloc_and_assign_all(stream, space, &blocks)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("AllocateBlocks failed: {}", e))
                    })?;
            }
            Command::DeallocateBlocks { stream, blocks } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .unassign_all(stream, space, &blocks)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("DeallocateBlocks failed: {}", e))
                    })?;
            }
            Command::FillBlock {
                stream,
                block,
                context,
                inputs,
                outputs,
            } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .fill(stream, space, block, context, inputs, outputs)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("FillBlock failed: {}", e))
                    })?;
            }
            Command::ExportBlocks {
                blocks,
                resource_name,
            } => {
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                let gids = self.driver_l4m.lookup_all(space, &blocks).map_err(|e| {
                    ControllerError::DriverError(format!("ExportBlocks lookup failed: {}", e))
                })?;
                self.exported_blocks
                    .insert(resource_name, ExportedBlocks::new(inst_id, gids));
            }
            Command::ImportBlocks {
                blocks,
                resource_name,
            } => {
                let space = *self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                let exported = self.exported_blocks.get(&resource_name).ok_or(
                    ControllerError::ExportedBlockNotFound(resource_name.clone()),
                )?;
                self.driver_l4m
                    .assign_all(&space, &blocks, &exported.addrs)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("ImportBlocks failed: {}", e))
                    })?;
            }
            Command::GetAllExportedBlocks { handle } => {
                let catalogue = self
                    .exported_blocks
                    .iter()
                    .map(|(k, v)| (k.clone(), v.addrs.len() as u32))
                    .collect();
                handle.send(catalogue).map_err(|_| {
                    ControllerError::SendError("GetAllExportedBlocks failed.".to_string())
                })?;
            }
            Command::CopyBlock {
                stream,
                src_block,
                dst_block,
                src_token_offset,
                dst_token_offset,
                size,
            } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .copy_tokens(
                        stream,
                        space,
                        src_block,
                        dst_block,
                        src_token_offset,
                        dst_token_offset,
                        size,
                    )
                    .map_err(|e| {
                        ControllerError::DriverError(format!("CopyBlock failed: {}", e))
                    })?;
            }

            Command::MaskBlock {
                stream,
                block,
                mask,
            } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .mask_tokens(stream, space, block, &mask)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("MaskBlock failed: {}", e))
                    })?;
            }
            Command::AllocateEmb { stream, embs } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .alloc_and_assign_all(stream, space, &embs)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("AllocateEmb failed: {}", e))
                    })?;
            }
            Command::DeallocateEmb { stream, embs } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .unassign_all(stream, space, &embs)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("DeallocateEmb failed: {}", e))
                    })?;
            }
            Command::EmbedText {
                stream,
                embs,
                text,
                positions,
            } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .embed_text(stream, space, embs, text, positions)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("EmbedText failed: {}", e))
                    })?;
            }
            Command::EmbedImage {
                stream,
                embs,
                image,
            } => {
                // let stream = Stream::new(&inst_id, Some(stream));
                // let space = *self
                //     .vspaces
                //     .get(&inst_id)
                //     .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                // self.driver_l4m
                //     .embed_img(stream, &space, embs, image)
                //     .map_err(|e| {
                //         ControllerError::DriverError(format!("EmbedImage failed: {}", e))
                //     })?;
            }
            Command::AllocateDist { stream, dists } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .alloc_and_assign_all(stream, space, &dists)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("AllocateDist failed: {}", e))
                    })?;
            }
            Command::DeallocateDist { stream, dists } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .unassign_all(stream, space, &dists)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("DeallocateDist failed: {}", e))
                    })?;
            }
            Command::DecodeTokenDist {
                stream,
                embs,
                dists,
            } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .next_token_dist(stream, space, embs, dists)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("DecodeTokenDist failed: {}", e))
                    })?;
            }
            Command::SampleTopK {
                stream,
                dist,
                k,
                handle,
            } => {
                let stream = Stream::new(&inst_id, Some(stream));
                let space = self
                    .vspaces
                    .get(&inst_id)
                    .ok_or(ControllerError::VspaceNotFound(inst_id))?;
                self.driver_l4m
                    .sample_top_k(stream, space, &dist, k, handle)
                    .map_err(|e| {
                        ControllerError::DriverError(format!("SampleTopK failed: {}", e))
                    })?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct ExportedBlocks {
    owner_id: InstanceId,
    addrs: Vec<object::Id<KvBlock>>,
}

impl ExportedBlocks {
    pub fn new(owner_id: InstanceId, addrs: Vec<object::Id<KvBlock>>) -> Self {
        Self { owner_id, addrs }
    }
}
