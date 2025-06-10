#![cfg(test)]

use axum::{
    body::Body,
    http::{self, Request, StatusCode},
    Router,
};
use crate::{
    handlers::{heartbeat_handler, list_backends_handler, register_backend_handler},
    models::{
        BackendInfoSummary, BackendRegistrationRequest, BackendRegistrationResponse, BackendStatus,
        HeartbeatResponse, ListBackendsResponse,
    },
    state::{AppState, SharedState},
};
use http_body_util::BodyExt; // for `collect`
use std::sync::{Arc, RwLock};
use tower::ServiceExt; // for `oneshot`
use uuid::Uuid;

fn app(state: SharedState) -> Router {
    Router::new()
        .route("/backends/register", axum::routing::post(register_backend_handler))
        .route("/backends/:backend_id/heartbeat", axum::routing::post(heartbeat_handler))
        .route("/backends", axum::routing::get(list_backends_handler))
        .with_state(state)
}

#[tokio::test]
async fn test_register_backend_success() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));

    let registration_req = BackendRegistrationRequest {
        capabilities: vec!["gpu".to_string(), "cuda-12".to_string()],
        management_api_address: "http://127.0.0.1:8080".to_string(),
    };

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri("/backends/register")
                .header(http::header::CONTENT_TYPE, mime::APPLICATION_JSON.as_ref())
                .body(Body::from(serde_json::to_string(&registration_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let reg_response: BackendRegistrationResponse = serde_json::from_slice(&body).unwrap();

    // Check that we got a valid UUID
    assert!(!reg_response.backend_id.to_string().is_empty());

    // Check state was updated
    let state = shared_state.read().unwrap();
    let backend_info = state.backends.get(&reg_response.backend_id).unwrap();
    assert_eq!(backend_info.status, BackendStatus::Initializing);
    assert_eq!(backend_info.management_api_address, registration_req.management_api_address);
    assert_eq!(backend_info.capabilities, registration_req.capabilities);
}

#[tokio::test]
async fn test_register_backend_empty_address() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));

    let registration_req = BackendRegistrationRequest {
        capabilities: vec!["cpu".to_string()],
        management_api_address: "".to_string(), // Empty address
    };

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri("/backends/register")
                .header(http::header::CONTENT_TYPE, mime::APPLICATION_JSON.as_ref())
                .body(Body::from(serde_json::to_string(&registration_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Check state was not updated
    let state = shared_state.read().unwrap();
    assert!(state.backends.is_empty());
}

#[tokio::test]
async fn test_heartbeat_success() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));
    
    // First, register a backend manually  
    let backend_id = {
        let mut state = shared_state.write().unwrap();
        let id = state.register_backend(
            vec!["test-cap".to_string()],
            "http://backend:1234".to_string(),
        );
        // Explicitly drop the lock
        drop(state);
        id
    };
    
    // Verify initial status
    {
        let state = shared_state.read().unwrap();
        assert_eq!(state.backends.get(&backend_id).unwrap().status, BackendStatus::Initializing);
        // Explicitly drop the lock
        drop(state);
    }

    // Make heartbeat request
    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let heartbeat_resp: HeartbeatResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(heartbeat_resp.message, "Heartbeat received");
    assert_eq!(heartbeat_resp.status_updated_to, Some(BackendStatus::Running));

    // Check state was updated
    {
        let state = shared_state.read().unwrap();
        let backend_info = state.backends.get(&backend_id).unwrap();
        assert_eq!(backend_info.status, BackendStatus::Running);
        assert!(backend_info.last_heartbeat.is_some());
        drop(state);
    }
}

#[tokio::test]
async fn test_heartbeat_unknown_backend() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));
    let unknown_backend_id = Uuid::new_v4();

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", unknown_backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_list_backends_empty() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::GET)
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let list_response: ListBackendsResponse = serde_json::from_slice(&body).unwrap();
    assert!(list_response.backends.is_empty());
}

#[tokio::test]
async fn test_list_backends_with_data() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));
    
    // Register a few backends manually
    let backend_id1 = {
        let mut state_guard = shared_state.write().unwrap();
        state_guard.register_backend(
            vec!["gpu".to_string()],
            "http://backend1:8080".to_string(),
        )
    };
    
    let backend_id2 = {
        let mut state_guard = shared_state.write().unwrap();
        state_guard.register_backend(
            vec!["cpu".to_string(), "avx2".to_string()],
            "http://backend2:8081".to_string(),
        )
    };

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::GET)
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let list_response: ListBackendsResponse = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(list_response.backends.len(), 2);
    
    // Find backends by ID
    let backend1_summary = list_response.backends.iter()
        .find(|b| b.backend_id == backend_id1)
        .expect("Backend 1 should be in the list");
    let backend2_summary = list_response.backends.iter()
        .find(|b| b.backend_id == backend_id2)
        .expect("Backend 2 should be in the list");
    
    assert_eq!(backend1_summary.status, BackendStatus::Initializing);
    assert_eq!(backend1_summary.capabilities, vec!["gpu".to_string()]);
    assert_eq!(backend1_summary.management_api_address, "http://backend1:8080");
    
    assert_eq!(backend2_summary.status, BackendStatus::Initializing);
    assert_eq!(backend2_summary.capabilities, vec!["cpu".to_string(), "avx2".to_string()]);
    assert_eq!(backend2_summary.management_api_address, "http://backend2:8081");
}

#[tokio::test]
async fn test_end_to_end_flow() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));

    // 1. Register a backend
    let registration_req = BackendRegistrationRequest {
        capabilities: vec!["gpu".to_string(), "cuda-12".to_string()],
        management_api_address: "http://127.0.0.1:8080".to_string(),
    };

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri("/backends/register")
                .header(http::header::CONTENT_TYPE, mime::APPLICATION_JSON.as_ref())
                .body(Body::from(serde_json::to_string(&registration_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let reg_response: BackendRegistrationResponse = serde_json::from_slice(&body).unwrap();
    let backend_id = reg_response.backend_id;

    // 2. Send heartbeat
    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // 3. List backends and verify the backend is there with Running status
    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::GET)
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let list_response: ListBackendsResponse = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(list_response.backends.len(), 1);
    let backend_summary = &list_response.backends[0];
    assert_eq!(backend_summary.backend_id, backend_id);
    assert_eq!(backend_summary.status, BackendStatus::Running);
    assert_eq!(backend_summary.capabilities, vec!["gpu".to_string(), "cuda-12".to_string()]);
    assert_eq!(backend_summary.management_api_address, "http://127.0.0.1:8080");
}
