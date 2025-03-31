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

// Import for request body deserialization
use serde::Deserialize;
use std::time::Instant;

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

// --- Define Request Structures ---
#[derive(Deserialize, Debug)]
struct RequestMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionRequest {
    model: String,
    input: Vec<RequestMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    // Other fields can be added as needed
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
        "/" | "/v1/responses" => handle_chat_completion(req, res).await,
        _ => handle_not_found(req, res).await,
    }
}

async fn handle_chat_completion(req: Request<IncomingBody>, res: Responder) -> Finished {
    println!("Processing chat completion request...");

    // Read and parse the request body
    let body_bytes = match req.into_body().bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Error reading request body: {:?}", e);
            return handle_error(res).await;
        }
    };

    let chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(e) => {
            eprintln!("Error parsing request JSON: {:?}", e);
            return handle_error(res).await;
        }
    };

    println!("Request for model: {}, with {} input",
             chat_request.model, chat_request.input.len());

    // Start timing for performance metrics
    let start = Instant::now();

    // Get available models and initialize
    let available_models = symphony::available_models();
    if available_models.is_empty() {
        eprintln!("No Symphony models available");
        return handle_error(res).await;
    }

    // Match the requested model
    let model_name = chat_request.model.as_str();
    if !available_models.contains(&model_name.to_string()) {
        eprintln!("Model {} not found", model_name);
        // Print available models for debugging
        for model in available_models.iter() {
            println!("Available model: {}", model);
        }
        return handle_error(res).await;
    }

    let model = match symphony::Model::new(model_name) {
        Some(model) => model,
        None => {
            eprintln!("Failed to create model");
            return handle_error(res).await;
        }
    };

    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    // Begin formatting the prompt
    ctx.fill("<|begin_of_text|>").await;

    // Add system message if present, otherwise use a default
    let mut has_system = false;
    for message in &chat_request.input {
        if message.role == "system" {
            ctx.fill(&format!("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", message.content)).await;
            has_system = true;
            break;
        }
    }

    // Add default system message if none was provided
    if !has_system {
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;
    }

    // Add user and assistant input in order
    for message in &chat_request.input {
        if message.role == "system" {
            continue; // Already handled system input
        }

        ctx.fill(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                         message.role, message.content)).await;
    }

    // Add the assistant prefix for generating the response
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n").await;

    // Determine max tokens to generate
    let max_tokens = chat_request.max_tokens.unwrap_or(256);
    println!("Generating response with max_tokens: {}", max_tokens);

    // Generate text until end token
    let generated_text = ctx.generate_until("<|eot_id|>", max_tokens as usize).await;
    let elapsed = start.elapsed();

    let token_ids = tokenizer.encode(&generated_text);
    println!("Generated {} tokens in {:?}", token_ids.len(), elapsed);

    // Create response
    let response_id = format!("chatcmpl-{}", generate_random_string(20));
    let timestamp = get_unix_timestamp();

    let response_data = OpenAiResponse {
        id: response_id,
        object: "chat.completion".to_string(),
        created: timestamp,
        model: chat_request.model.clone(),
        choices: vec![
            ResponseChoice {
                index: 0,
                message: ResponseChoiceMessage {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
                finish_reason: "stop".to_string(),
            }
        ],
    };

    // Serialize and send response
    let response_body_json = match serde_json::to_string(&response_data) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("Error serializing response JSON: {}", e);
            return handle_error(res).await;
        }
    };

    println!("Sending response: {} characters generated in {:?}",
             response_data.choices[0].message.content.len(), elapsed);

    // Create response with proper headers
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