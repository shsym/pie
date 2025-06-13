#![cfg(test)]

use axum::{
    body::Body,
    http::{self, Request, StatusCode},
    Router,
};
use crate::{
    handlers::{heartbeat_handler, list_backends_handler, register_backend_handler, terminate_backend_handler}, // Added terminate_backend_handler
    models::{
        BackendRegistrationRequest, BackendRegistrationResponse, BackendStatus,
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
        .route("/backends/:backend_id/terminate", axum::routing::post(terminate_backend_handler)) // Added terminate route
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
async fn test_register_backend_empty_capabilities() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));

    let registration_req = BackendRegistrationRequest {
        capabilities: vec![], // Empty capabilities
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Check state was not updated
    let state = shared_state.read().unwrap();
    assert!(state.backends.is_empty());
}

#[tokio::test]
async fn test_register_backend_invalid_management_address_format() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));

    let registration_req = BackendRegistrationRequest {
        capabilities: vec!["cpu".to_string()],
        management_api_address: "this-is-not-a-valid-url".to_string(), // Invalid address
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
    let backend_id;

    // First, register a backend manually
    {
        let mut state = shared_state.write().unwrap();
        backend_id = state.register_backend(
            vec!["test-cap".to_string()],
            "http://backend:1234".to_string(),
        );
        drop(state);
    }

    // Verify initial status
    let initial_heartbeat_time;
    {
        let state = shared_state.read().unwrap();
        assert_eq!(state.backends.get(&backend_id).unwrap().status, BackendStatus::Initializing);
        initial_heartbeat_time = state.backends.get(&backend_id).unwrap().last_heartbeat;
        assert!(initial_heartbeat_time.is_none()); // No heartbeat yet
        drop(state);
    }

    // Make first heartbeat request
    let response1 = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response1.status(), StatusCode::OK);
    let body1 = response1.into_body().collect().await.unwrap().to_bytes();
    let heartbeat_resp1: HeartbeatResponse = serde_json::from_slice(&body1).unwrap();
    assert_eq!(heartbeat_resp1.message, "Heartbeat received");
    assert_eq!(heartbeat_resp1.status_updated_to, Some(BackendStatus::Running));

    // Check state was updated after first heartbeat
    let first_heartbeat_time;
    {
        let state = shared_state.read().unwrap();
        let backend_info = state.backends.get(&backend_id).unwrap();
        assert_eq!(backend_info.status, BackendStatus::Running);
        first_heartbeat_time = backend_info.last_heartbeat;
        assert!(first_heartbeat_time.is_some());
        drop(state);
    }

    // Make a small delay to ensure timestamp changes
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    // Make second heartbeat request
    let response2 = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response2.status(), StatusCode::OK);
    let body2 = response2.into_body().collect().await.unwrap().to_bytes();
    let heartbeat_resp2: HeartbeatResponse = serde_json::from_slice(&body2).unwrap();
    assert_eq!(heartbeat_resp2.message, "Heartbeat received");
    assert_eq!(heartbeat_resp2.status_updated_to, None); // Status should not change from Running

    // Check state was updated after second heartbeat
    {
        let state = shared_state.read().unwrap();
        let backend_info = state.backends.get(&backend_id).unwrap();
        assert_eq!(backend_info.status, BackendStatus::Running); // Still running
        assert!(backend_info.last_heartbeat.is_some());
        assert_ne!(backend_info.last_heartbeat, first_heartbeat_time, "Last heartbeat time should have updated");
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
async fn test_terminate_backend_success() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));
    let backend_id;

    // 1. Register a backend
    {
        let mut state = shared_state.write().unwrap();
        backend_id = state.register_backend(
            vec!["test-cap".to_string()],
            "http://backend-to-terminate:5678".to_string(),
        );
        drop(state);
    }

    // 2. Send heartbeat to make it Running
    let _ = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    {
        let state = shared_state.read().unwrap();
        assert_eq!(state.backends.get(&backend_id).unwrap().status, BackendStatus::Running);
        drop(state);
    }

    // 3. Send terminate request
    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/terminate", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // Assuming terminate returns a simple message or empty body on success
    // let body = response.into_body().collect().await.unwrap().to_bytes();
    // let terminate_response: SomeResponseType = serde_json::from_slice(&body).unwrap();
    // assert_eq!(terminate_response.message, "Termination signal sent");


    // 4. Check state was updated (e.g., to Terminated)
    //    Note: The actual status might depend on the implementation of terminate_backend_handler
    //    and whether it's synchronous or asynchronous. Assuming it updates to Terminated.
    {
        let state = shared_state.read().unwrap();
        let backend_info = state.backends.get(&backend_id).unwrap();
        assert_eq!(backend_info.status, BackendStatus::Terminated); // Or Terminating, then Terminated
        drop(state);
    }
}

#[tokio::test]
async fn test_terminate_backend_not_found() {
    let shared_state = Arc::new(RwLock::new(AppState::new()));
    let unknown_backend_id = Uuid::new_v4();

    let response = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/terminate", unknown_backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
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
    let response_heartbeat = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/heartbeat", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response_heartbeat.status(), StatusCode::OK);

    // 3. List backends and verify the backend is there with Running status
    let response_list_running = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::GET)
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response_list_running.status(), StatusCode::OK);
    let body_list_running = response_list_running.into_body().collect().await.unwrap().to_bytes();
    let list_response_running: ListBackendsResponse = serde_json::from_slice(&body_list_running).unwrap();

    assert_eq!(list_response_running.backends.len(), 1);
    let backend_summary_running = &list_response_running.backends[0];
    assert_eq!(backend_summary_running.backend_id, backend_id);
    assert_eq!(backend_summary_running.status, BackendStatus::Running);
    assert_eq!(backend_summary_running.capabilities, vec!["gpu".to_string(), "cuda-12".to_string()]);
    assert_eq!(backend_summary_running.management_api_address, "http://127.0.0.1:8080");

    // 4. Terminate the backend
    let response_terminate = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(format!("/backends/{}/terminate", backend_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response_terminate.status(), StatusCode::OK);

    // 5. List backends again and verify the backend is in Stopped status
    let response_list_stopped = app(shared_state.clone())
        .oneshot(
            Request::builder()
                .method(http::Method::GET)
                .uri("/backends")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response_list_stopped.status(), StatusCode::OK);
    let body_list_stopped = response_list_stopped.into_body().collect().await.unwrap().to_bytes();
    let list_response_stopped: ListBackendsResponse = serde_json::from_slice(&body_list_stopped).unwrap();

    assert_eq!(list_response_stopped.backends.len(), 1);
    let backend_summary_stopped = &list_response_stopped.backends[0];
    assert_eq!(backend_summary_stopped.backend_id, backend_id);
    assert_eq!(backend_summary_stopped.status, BackendStatus::Terminated); // Verify status after termination
    assert_eq!(backend_summary_stopped.capabilities, vec!["gpu".to_string(), "cuda-12".to_string()]);
    assert_eq!(backend_summary_stopped.management_api_address, "http://127.0.0.1:8080");
}
