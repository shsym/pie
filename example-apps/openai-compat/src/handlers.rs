use inferlet::wstd::http::body::IncomingBody;
use inferlet::wstd::http::server::{Finished, Responder};
use inferlet::wstd::http::{IntoBody, Request, Response, StatusCode};
use serde_json;
use std::time::Instant;
use inferlet::traits::Tokenize;
use crate::models::{
    ChatCompletionRequest, CompletionTokensDetails, OpenAiResponse, ResponseChoice,
    ResponseChoiceMessage, TokensDetails, UsageStats,
};
use crate::utils::{generate_random_string, get_unix_timestamp};

/// Handle chat completion requests
pub async fn handle_chat_completion(req: Request<IncomingBody>, res: Responder) -> Finished {
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

    println!(
        "Request for model: {}, with {} messages",
        chat_request.model,
        chat_request.messages.len()
    );

    // Start timing for performance metrics
    let start = Instant::now();

    let model = inferlet::get_auto_model();

    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    // Begin formatting the prompt
    ctx.fill("<|begin_of_text|>");

    // Add system message if present, otherwise use a default
    let mut has_system = false;
    for message in &chat_request.messages {
        if message.role == "system" {
            ctx.fill(&format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                message.content
            ));
            has_system = true;
            break;
        }
    }

    // Add default system message if none was provided
    if !has_system {
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    }

    // Add user and assistant messages in order
    for message in &chat_request.messages {
        if message.role == "system" {
            continue; // Already handled system messages
        }

        ctx.fill(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            message.role, message.content
        ));
    }

    // Add the assistant prefix for generating the response
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    // Determine max tokens to generate
    let max_tokens = chat_request.max_tokens.unwrap_or(256);
    println!("Generating response with max_tokens: {}", max_tokens);

    // Generate text until end token
    let generated_text = ctx.generate_until(max_tokens as usize).await;
    let elapsed = start.elapsed();

    let token_ids = tokenizer.tokenize(&generated_text);
    println!("Generated {} tokens in {:?}", token_ids.len(), elapsed);

    // Create response
    let response_id = format!("chatcmpl-{}", generate_random_string(20));
    let timestamp = get_unix_timestamp();

    // Calculate token counts
    let prompt_tokens = chat_request
        .messages
        .iter()
        .map(|m| tokenizer.tokenize(&m.content).len())
        .sum::<usize>();
    let completion_tokens = token_ids.len();
    let total_tokens = prompt_tokens + completion_tokens;

    let response_data = OpenAiResponse {
        id: response_id,
        object: "chat.completion".to_string(),
        created: timestamp,
        model: chat_request.model.clone(),
        choices: vec![ResponseChoice {
            index: 0,
            message: ResponseChoiceMessage {
                role: "assistant".to_string(),
                content: generated_text,
                refusal: None,
                annotations: vec![],
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStats {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            total_tokens: total_tokens as u32,
            prompt_tokens_details: TokensDetails {
                cached_tokens: 0,
                audio_tokens: 0,
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: completion_tokens as u32,
                audio_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            },
        },
        service_tier: "default".to_string(),
    };

    // Serialize and send response
    let response_body_json = match serde_json::to_string(&response_data) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("Error serializing response JSON: {}", e);
            return handle_error(res).await;
        }
    };

    println!(
        "Sending response: {} characters generated in {:?}",
        response_data.choices[0].message.content.len(),
        elapsed
    );

    // Create response with proper headers
    let response = Response::builder()
        .header("Content-Type", "application/json")
        .body(response_body_json.into_body())
        .unwrap();

    // Send the response
    res.respond(response).await
}

/// Handle 404 Not Found responses
pub async fn handle_not_found(_req: Request<IncomingBody>, res: Responder) -> Finished {
    let body = "Not Found: The requested endpoint does not exist.";
    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "text/plain")
        .body(body.into_body())
        .unwrap();

    res.respond(response).await
}

/// Handle error responses
pub async fn handle_error(res: Responder) -> Finished {
    let body = r#"{"error": {"message": "An error occurred while processing the request", "type": "internal_error"}}"#;
    let response = Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header("Content-Type", "application/json")
        .body(body.into_body())
        .unwrap();

    res.respond(response).await
}
