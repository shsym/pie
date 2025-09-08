use futures::future::join_all;
use futures::{StreamExt, stream::FuturesUnordered};
use inferlet::{Context, Model};
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

// --- Help Message and Constants ---
const HELP: &str = "\
Usage: graph-of-thought [OPTIONS]

A program to test hierarchical aggregation by generating and combining multiple proposals for a given question.

Options:
  --question <QUESTION>          The question to process [default: What is the sum of 123456789 and 987654321?]
  --proposal-tokens <TOKENS>     Comma-separated list of max tokens for initial proposals [default: 4,32,16,8,4,16,3,32]
  --aggregation-tokens <TOKENS>  Max tokens for each aggregation step [default: 128]
  -h, --help                     Prints this help message";

const SYSTEM_PROMPT: &str = "You are a helpful, respectful and honest assistant.";
const PROPOSAL_PROMPT_TEMPLATE: &str = "Could you suggest a method or approach to solve the following question? Please provide a high-level plan without doing the actual calculation. Keep it concise, around 80 words. Question: {}";
const AGGREGATE_PROMPT: &str = "Please compare the following solution with the one you just provided and aggregate their ideas into a single, improved solution:\n";

// --- Llama-3.1 Instruct Chat Template Tokens ---
const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOT_ID: &str = "<|eot_id|>";
const SYSTEM_HEADER: &str = "<|start_header_id|>system<|end_header_id|>\n\n";
const USER_HEADER: &str = "<|start_header_id|>user<|end_header_id|>\n\n";
const ASSISTANT_HEADER: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";

/// Main logic for running the hierarchical aggregation workflow.
async fn run_hierarchical_aggregation(
    mut base_context: Context,
    question: &str,
    proposal_tokens: Vec<usize>,
    aggregation_tokens: usize,
) -> Vec<String> {
    // --- Stage 1: Generate Initial Proposals ---
    // Create a base prompt for all proposals, then fork it for each generation task.
    let propose_prompt = format!(
        "{}{}{}{}{}{}",
        USER_HEADER,
        PROPOSAL_PROMPT_TEMPLATE.replace("{}", question),
        EOT_ID,
        ASSISTANT_HEADER,
        "", // Assistant response starts here
        ""
    );
    base_context.fill(&propose_prompt);
    base_context.flush(); // Ensure the base prompt is processed before forking

    let mut proposal_tasks = proposal_tokens
        .into_iter()
        .map(|max_tokens| {
            let mut ctx = base_context.fork();
            async move {
                let proposal_text = ctx.generate_until(max_tokens).await;
                (proposal_text, ctx)
            }
        })
        .collect::<FuturesUnordered<_>>();

    // --- Stage 2: First-Level Aggregation (Pairing Proposals) ---
    // As proposals complete, pair them up and generate a combined solution.
    let mut first_aggregation_tasks = FuturesUnordered::new();
    let mut pending_proposal: Option<(String, Context)> = None;

    while let Some((proposal_text, mut proposal_ctx)) = proposal_tasks.next().await {
        if pending_proposal.is_none() {
            pending_proposal = Some((proposal_text, proposal_ctx));
        } else {
            let (previous_proposal_text, _) = pending_proposal.take().unwrap();
            let aggregation_prompt = format!(
                "{}{}{}{}{}{}{}",
                EOT_ID, // End previous assistant turn
                USER_HEADER,
                AGGREGATE_PROMPT,
                previous_proposal_text,
                EOT_ID,
                ASSISTANT_HEADER,
                ""
            );
            proposal_ctx.fill(&aggregation_prompt);
            first_aggregation_tasks.push(async move {
                let aggregation_text = proposal_ctx
                    .generate_until(aggregation_tokens)
                    .await;
                (aggregation_text, proposal_ctx)
            });
        }
    }

    // --- Stage 3: Second-Level Aggregation (Pairing Aggregations) ---
    // Pair the results from the first aggregation level to create the final solutions.
    let mut second_aggregation_tasks = Vec::new();
    let mut pending_aggregation: Option<(String, Context)> = None;

    while let Some((aggregation_text, mut aggregation_ctx)) = first_aggregation_tasks.next().await {
        if pending_aggregation.is_none() {
            pending_aggregation = Some((aggregation_text, aggregation_ctx));
        } else {
            let (previous_aggregation_text, _) = pending_aggregation.take().unwrap();
            let final_prompt = format!(
                "{}{}{}{}{}{}{}",
                EOT_ID, // End previous assistant turn
                USER_HEADER,
                AGGREGATE_PROMPT,
                previous_aggregation_text,
                EOT_ID,
                ASSISTANT_HEADER,
                ""
            );
            aggregation_ctx.fill(&final_prompt);
            second_aggregation_tasks.push(async move {
                aggregation_ctx
                    .generate_until(aggregation_tokens)
                    .await
            });
        }
    }

    // --- Stage 4: Collect Final Results ---
    join_all(second_aggregation_tasks).await
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- 1. Argument Parsing ---
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let question: String = args
        .opt_value_from_str("--question")
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "What is the sum of 123456789 and 987654321?".to_string());

    let proposal_tokens_str: String = args
        .opt_value_from_str("--proposal-tokens")
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "4,32,16,8,4,16,3,32".to_string());

    let proposal_tokens: Vec<usize> = proposal_tokens_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    let aggregation_tokens: usize = args
        .opt_value_from_str("--aggregation-tokens")
        .map_err(|e| e.to_string())?
        .unwrap_or(128);

    let start = Instant::now();
    println!(
        "--- Starting hierarchical aggregation for question: \"{}\" ---",
        question
    );
    println!(
        "Proposal tokens: {:?}, Aggregation tokens: {}",
        proposal_tokens, aggregation_tokens
    );

    // --- 2. Initialize Model and Root Context ---
    let model = inferlet::get_auto_model();
    let mut ctx_root = model.create_context();
    let system_setup = format!(
        "{}{}{}{}{}",
        BOS_TOKEN, SYSTEM_HEADER, SYSTEM_PROMPT, EOT_ID, ""
    );
    ctx_root.fill(&system_setup);

    // --- 3. Run Aggregation Logic ---
    let final_solutions =
        run_hierarchical_aggregation(ctx_root, &question, proposal_tokens, aggregation_tokens)
            .await;

    println!("\n--- Aggregation complete in {:?} ---\n", start.elapsed());

    // --- 4. Print Results ---
    for (i, solution) in final_solutions.iter().enumerate() {
        println!("Final aggregated solution #{}:\n{}\n", i + 1, solution);
    }
    Ok(())
}
