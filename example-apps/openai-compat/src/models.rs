use serde::{Serialize, Deserialize};
use serde_json;

// --- Request Structures ---
#[derive(Deserialize, Debug)]
pub struct RequestMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    // Other fields can be added as needed
}

// --- Response Structures (OpenAI Format) ---
#[derive(Serialize, Debug)]
pub struct TokensDetails {
    pub cached_tokens: u32,
    pub audio_tokens: u32,
}

#[derive(Serialize, Debug)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub audio_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

#[derive(Serialize, Debug)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: TokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Serialize, Debug)]
pub struct ResponseChoiceMessage {
    pub role: String,
    pub content: String,
    pub refusal: Option<String>,
    pub annotations: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct ResponseChoice {
    pub index: u32,
    pub message: ResponseChoiceMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Serialize, Debug)]
pub struct OpenAiResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ResponseChoice>,
    pub usage: UsageStats,
    pub service_tier: String,
}