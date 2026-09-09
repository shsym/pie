use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use async_trait::async_trait;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use bytes::Bytes;
use worker_api::BlobRef;

#[derive(Clone, Debug)]
pub struct Blob {
    pub kind: String,
    pub bytes: Bytes,
}

#[async_trait]
pub trait BlobStore: Send + Sync + 'static {
    async fn put(&self, kind: String, bytes: Bytes) -> Result<BlobRef>;

    async fn get(&self, hash: &str) -> Result<Option<Blob>>;
}

pub struct GatewayOriginStore {
    origin: String,
    blobs: RwLock<HashMap<String, Blob>>,
}

impl GatewayOriginStore {
    pub fn new(origin: impl Into<String>) -> Self {
        Self {
            origin: origin.into(),
            blobs: RwLock::new(HashMap::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.blobs.read().expect("blob store lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl BlobStore for GatewayOriginStore {
    async fn put(&self, kind: String, bytes: Bytes) -> Result<BlobRef> {
        let hash = blake3::hash(&bytes).to_hex().to_string();
        let size = bytes.len() as u64;
        self.blobs
            .write()
            .expect("blob store lock poisoned")
            .entry(hash.clone())
            .or_insert_with(|| Blob {
                kind: kind.clone(),
                bytes,
            });
        Ok(BlobRef {
            hash,
            size,
            kind,
            origin: self.origin.clone(),
        })
    }

    async fn get(&self, hash: &str) -> Result<Option<Blob>> {
        Ok(self
            .blobs
            .read()
            .expect("blob store lock poisoned")
            .get(hash)
            .cloned())
    }
}

pub fn router(store: Arc<dyn BlobStore>) -> Router {
    Router::new()
        .route("/blob/{hash}", get(serve_blob))
        .with_state(store)
}

async fn serve_blob(State(store): State<Arc<dyn BlobStore>>, Path(hash): Path<String>) -> Response {
    match store.get(&hash).await {
        Ok(Some(blob)) => {
            let content_type = HeaderValue::from_str(&blob.kind)
                .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream"));
            ([(header::CONTENT_TYPE, content_type)], blob.bytes).into_response()
        }
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::warn!(%hash, error = %e, "blob fetch failed");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}
