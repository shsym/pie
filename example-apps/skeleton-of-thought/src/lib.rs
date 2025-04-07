use futures::future::join_all;
use symphony::{Context, Model};

// Constants with improved prompt templates
const PLAN_PROMPT_TEMPLATE: &str = "Generate up to {} key points that outline the answer to the following question: {}. Each point should be a concise statement of a main idea, enclosed in <point> tags. Do not elaborate on the points yet. Keep your entire response within 30 words.";
const COMPLETE_PROMPT_TEMPLATE: &str = "Elaborate on the following point: {}. Your response should be complete and only concerned with this point. Keep it concise and within 80 words.";
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";
const MAX_TOKENS: usize = 32;

/// Generates a high-level plan and elaborates on each point in parallel.
///
/// # Arguments
/// - `ctx`: The initial context containing the system prompt.
/// - `question`: The question to answer.
/// - `max_points`: The maximum number of points to generate in the plan.
///
/// # Returns
/// A vector of elaborated responses, one for each point.
async fn plan_and_generate_parallel(
    mut ctx: Context,
    question: &str,
    max_points: usize,
) -> Vec<String> {
    // Fork a context for generating the plan, preserving the original ctx
    let mut plan_ctx = ctx.fork();
    let plan_prompt = format!("{} {} {}", PLAN_PROMPT_TEMPLATE, max_points, question);
    plan_ctx.fill(&plan_prompt);
    plan_ctx.fill(ASSISTANT_PREFIX);
    let output = plan_ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

    // Robustly parse points from the output
    let points = output
        .split("<point>")
        .skip(1) // Skip any text before the first "<point>"
        .filter_map(|s| {
            let trimmed = s.trim();
            if let Some(end) = trimmed.find("</point>") {
                Some(trimmed[..end].to_string())
            } else {
                None // Ignore incomplete or malformed tags
            }
        })
        .collect::<Vec<_>>();

    // Generate elaborations in parallel using fresh contexts
    let leaf_futures = points
        .into_iter()
        .map(|point| {
            let mut elab_ctx = ctx.fork(); // Fork from original ctx for clean state
            let complete_prompt = format!("{} {}", COMPLETE_PROMPT_TEMPLATE, point);
            elab_ctx.fill(&complete_prompt);
            elab_ctx.fill(ASSISTANT_PREFIX);
            async move { elab_ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await }
        })
        .collect::<Vec<_>>();

    join_all(leaf_futures).await
}

#[symphony::main]
async fn main() -> Result<(), String> {
    // Initialize the model and context
    let available_models = symphony::available_models();
    let model = Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    // Define the question and parameters
    let question = "What are the defining characteristics of Rome?";
    let max_points = 3;

    // Execute and display results
    let elaborations = plan_and_generate_parallel(ctx, question, max_points).await;
    for (i, elaboration) in elaborations.iter().enumerate() {
        println!("Elaboration {}: {}", i + 1, elaboration);
    }

    Ok(())
}
