// Replace TCP socket handling with HTTP server imports
use symphony::wstd::http::body::IncomingBody;
use symphony::wstd::http::server::{Finished, Responder};
use symphony::wstd::http::{IntoBody, Request, Response, StatusCode};

// Added imports
use serde::Serialize; // For serializing the response structure
use serde_json;      // For converting the struct to a JSON string
use rand::{rng, Rng}; // For random generation
use rand_distr::Alphanumeric;
use std::time::{SystemTime, UNIX_EPOCH}; // For timestamp

// --- Define Response Structures (OpenAI Format) ---
#[derive(Serialize, Debug)]
struct ResponseChoiceMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct ResponseChoice {
    index: u32,
    message: ResponseChoiceMessage,
    finish_reason: String,
}

#[derive(Serialize, Debug)]
struct OpenAiResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ResponseChoice>,
    // usage: Option<UsageStats>, // We can omit usage for now
}

// --- Helper Function for Random String ---
fn generate_random_string(len: usize) -> String {
    rng()
        .sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

// --- Helper Function for Timestamp ---
fn get_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default() // Handle potential time errors simply
        .as_secs()
}

#[symphony::server_main]
async fn main(req: Request<IncomingBody>, res: Responder) -> Finished {
    // Log connection info
    println!("Received request from {} to {}", 
        req.headers().get("host").map_or("unknown", |h| h.to_str().unwrap_or("unknown")),
        req.uri().path_and_query().map_or("", |p| p.as_str()));

    match req.uri().path() {
        "/" | "/v1/chat/completions" => handle_chat_completion(req, res).await,
        _ => handle_not_found(req, res).await,
    }
}

async fn handle_chat_completion(_req: Request<IncomingBody>, res: Responder) -> Finished {
    // --- Generate Response Content ---
    let random_content = generate_random_string(30); // Generate a 30-char random string
    let response_id = format!("chatcmpl-{}", generate_random_string(20)); // Generate a plausible ID
    let timestamp = get_unix_timestamp();

    // --- Construct the OpenAI-like Response Body ---
    let response_data = OpenAiResponse {
        id: response_id,
        object: "chat.completion".to_string(),
        created: timestamp,
        model: "symphony-mock-v1".to_string(),
        choices: vec![
            ResponseChoice {
                index: 0,
                message: ResponseChoiceMessage {
                    role: "assistant".to_string(),
                    content: random_content, // Embed the random string
                },
                finish_reason: "stop".to_string(),
            }
        ],
    };

    // --- Serialize Response to JSON ---
    let response_body_json = match serde_json::to_string(&response_data) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("Error serializing response JSON: {}", e);
            return handle_error(res).await;
        }
    };

    println!("Sending JSON response: {} bytes", response_body_json.len());

    // Create a response with proper headers
    let response = Response::builder()
        .header("Content-Type", "application/json")
        .body(response_body_json.into_body())
        .unwrap();

    // Send the response
    res.respond(response).await
}

async fn handle_not_found(_req: Request<IncomingBody>, res: Responder) -> Finished {
    let body = "Not Found: The requested endpoint does not exist.";
    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "text/plain")
        .body(body.into_body())
        .unwrap();

    res.respond(response).await
}

async fn handle_error(res: Responder) -> Finished {
    let body = r#"{"error": {"message": "An error occurred while processing the request", "type": "internal_error"}}"#;
    let response = Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header("Content-Type", "application/json")
        .body(body.into_body())
        .unwrap();

    res.respond(response).await
}