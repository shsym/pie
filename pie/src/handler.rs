use crate::batching::BatchingConfig;
use crate::model::HandlerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

pub mod core;
pub mod forward;
pub mod image;

pub mod adapter;
pub mod evolve;
pub mod tokenize;

// big message table

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Handler {
    Handshake,
    Query,
    Synchronize,
    Heartbeat,
    ForwardPass,
    EmbedImage,
    InitializeAdapter,
    UpdateAdapter,
    UploadAdapter,
    DownloadAdapter,
}

impl Handler {
    pub fn get_handler_id(&self) -> HandlerId {
        match self {
            Self::Handshake => 0,
            Self::Heartbeat => 1,
            Self::Synchronize => 0,
            Self::Query => 2,
            Self::ForwardPass => 3,
            Self::EmbedImage => 4,
            Self::InitializeAdapter => 5,
            Self::UpdateAdapter => 6,
            Self::UploadAdapter => 7,
            Self::DownloadAdapter => 8,
        }
    }
}

pub fn get_batching_config() -> HashMap<Handler, BatchingConfig> {
    let mut config = HashMap::new();
    config.insert(
        Handler::Synchronize,
        BatchingConfig::Bounded {
            max_wait_time: Duration::ZERO,
            min_size: 1,
            max_size: None,
        },
    );
    config.insert(
        Handler::ForwardPass,
        BatchingConfig::Triggered {
            trigger: None,
            min_wait_time: Duration::ZERO,
        },
    );
    config
}
