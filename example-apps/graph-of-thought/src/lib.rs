use futures::future::join_all;
use futures::{StreamExt, stream::FuturesUnordered};
use inferlet::{Context, Model};

const PROPOSAL_PROMPT: &str = "Could you suggest a method or approach to solve the following question? Please provide a high-level plan without doing the actual calculation. Keep it concise, around 80 words. Question: {}";
const AGGREGATE_PROMPT: &str = "Please compare the following solution with the one you just provided and aggregate their ideas into a single, improved solution:\n";
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";

// Main function to aggregate proposals asynchronously
async fn aggregate_proposals_async(mut base_context: Context, question: &str) -> Vec<String> {
    // Prepare the prompt for generating proposals
    let propose_prompt = format!("{} {}", PROPOSAL_PROMPT, question);
    base_context.fill(&propose_prompt);
    base_context.fill(ASSISTANT_PREFIX);
    base_context.flush();
    // Generate proposals in parallel with varying max_tokens
    let mut proposal_tasks = [4, 32, 16, 8, 4, 16, 3, 32]
        .into_iter()
        .map(|max_tokens| {
            let mut ctx = base_context.fork_unsafe();
            async move {
                let proposal_text = ctx.generate_until(STOP_TOKEN, max_tokens).await;
                (proposal_text, ctx)
            }
        })
        .collect::<FuturesUnordered<_>>();

    // First level of aggregation: pair proposals as they complete
    let mut first_aggregation_tasks = FuturesUnordered::<_>::new();
    let mut pending_proposal = None;

    while let Some((proposal_text, mut proposal_ctx)) = proposal_tasks.next().await {
        if pending_proposal.is_none() {
            pending_proposal = Some(proposal_text);
        } else {
            let previous_proposal = pending_proposal.take().unwrap();
            proposal_ctx.fill(AGGREGATE_PROMPT);
            proposal_ctx.fill(&previous_proposal);
            proposal_ctx.fill(ASSISTANT_PREFIX);
            first_aggregation_tasks.push(async move {
                let aggregation_text = proposal_ctx.generate_until(STOP_TOKEN, 32).await;
                (aggregation_text, proposal_ctx)
            });
        }
    }

    // Second level of aggregation: pair first-level aggregations
    let mut second_aggregation_tasks = Vec::new();
    let mut pending_aggregation = None;

    while let Some((aggregation_text, mut aggregation_ctx)) = first_aggregation_tasks.next().await {
        if pending_aggregation.is_none() {
            pending_aggregation = Some(aggregation_text);
        } else {
            let previous_aggregation = pending_aggregation.take().unwrap();
            aggregation_ctx.fill(AGGREGATE_PROMPT);
            aggregation_ctx.fill(&previous_aggregation);
            aggregation_ctx.fill(ASSISTANT_PREFIX);
            second_aggregation_tasks
                .push(async move { aggregation_ctx.generate_until(STOP_TOKEN, 32).await });
        }
    }

    // Collect and return the final aggregated solutions
    join_all(second_aggregation_tasks).await
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    // Initialize the model and context
    let available_models = inferlet::available_models();
    let model = Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");

    // Example usage
    let question = "What is the sum of 123456789 and 987654321?";
    let final_solutions = aggregate_proposals_async(ctx, question).await;

    // Print the final aggregated solutions for demonstration
    for (i, solution) in final_solutions.iter().enumerate() {
        println!("Final aggregated solution {}: {}", i + 1, solution);
    }
    Ok(())
}
