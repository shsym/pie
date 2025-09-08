use futures::future::join_all;
use inferlet::Context;
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

// --- Constants ---
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";

const HELP: &str = "
Usage: plan_and_generate_parallel [OPTIONS]

A program that first generates a plan (a list of key points) for a question, 
and then elaborates on each point concurrently.

Options:
  -p, --num-points <POINTS>  Sets the maximum number of key points to generate in the plan [default: 3]
  -t, --plan-tokens <TOKENS> Sets the max tokens for the planning generation [default: 80]
  -e, --elab-tokens <TOKENS> Sets the max tokens for each elaboration generation [default: 80]
  -h, --help                 Prints this help message
";

/// Generates a high-level plan and elaborates on each point in parallel.
async fn plan_and_generate_parallel(
    ctx: Context, // `ctx` is consumed, as it's only used for forking.
    question: &str,
    max_points: usize,
    plan_max_tokens: usize,
    elab_max_tokens: usize,
) -> Vec<String> {
    // 1. --- Planning Stage ---
    // Fork a context for generating the plan.
    let mut plan_ctx = ctx.fork();
    let plan_prompt = format!(
        "Generate up to {} key points that outline the answer to the following question: {}. Each point must be enclosed in <point> tags.",
        max_points, question
    );
    plan_ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{}{}",
        plan_prompt, STOP_TOKEN
    ));
    plan_ctx.fill(ASSISTANT_PREFIX);
    let output = plan_ctx.generate_until(plan_max_tokens).await;

    // 2. --- Point Parsing ---
    // Robustly parse points from the output.
    let points: Vec<String> = output
        .split("<point>")
        .skip(1)
        .filter_map(|s| s.split("</point>").next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if points.is_empty() {
        return Vec::new();
    }

    // 3. --- Elaboration Stage (Parallel) ---
    // Fork from the original base context for a clean state for each elaboration.
    let leaf_futures = points
        .into_iter()
        .map(|point| {
            let mut elab_ctx = ctx.fork();
            let complete_prompt = format!("Elaborate on the following point: {}. Your response should be complete and only concerned with this point.", point);
            elab_ctx.fill(&format!("<|start_header_id|>user<|end_header_id|>\n\n{}{}", complete_prompt, STOP_TOKEN));
            elab_ctx.fill(ASSISTANT_PREFIX);
            async move { elab_ctx.generate_until(elab_max_tokens).await }
        })
        .collect::<Vec<_>>();

    join_all(leaf_futures).await
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- Argument Parsing ---
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
    let num_points: usize = args
        .opt_value_from_str(["-p", "--num-points"])
        .map_err(|e| e.to_string())?
        .unwrap_or(3);
    let plan_max_tokens: usize = args
        .opt_value_from_str(["-t", "--plan-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(80);
    let elab_max_tokens: usize = args
        .opt_value_from_str(["-e", "--elab-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(80);

    let start = Instant::now();

    // --- Setup ---
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    let question = "What are the defining characteristics of Rome?";

    // --- Execute ---
    println!(
        "--- Starting plan and generate (plan: {} points, {} tokens; elab: {} tokens) ---",
        num_points, plan_max_tokens, elab_max_tokens
    );
    let elaborations =
        plan_and_generate_parallel(ctx, question, num_points, plan_max_tokens, elab_max_tokens)
            .await;

    println!("\n--- Completed in {:?} ---\n", start.elapsed());

    // --- Print Results ---
    if elaborations.is_empty() {
        println!("No points were generated or elaborated upon.");
    } else {
        for (i, elaboration) in elaborations.iter().enumerate() {
            println!("Elaboration {}:\n{}\n", i + 1, elaboration);
        }
    }

    Ok(())
}
