use futures::future::join_all;
use inferlet::{Context, Model};
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

// --- Prompt Templates & Chat Structure ---
const SYSTEM_PROMPT: &str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful, and honest assistant that excels at mathematical reasoning. Please follow the user's instructions precisely.<|eot_id|>";
const USER_HEADER: &str = "<|start_header_id|>user<|end_header_id|>\n\n";
const ASSISTANT_HEADER: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const EOT_ID: &str = "<|eot_id|>";

// --- Task-Specific Prompts ---
const PROPOSE_PROMPT_TEMPLATE: &str = "Please generate a high-level plan for solving the following question. First, just state the method you will use. Do not do the actual calculation. Keep your response concise and within 80 words. Question: ";
const EXECUTE_PROMPT: &str = "The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the plan. Give me the final answer. Make your response short.";
const REFLECT_PROMPT: &str = "Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. Please rigorously check the correctness of the calculations and the final answer.";

const HELP: &str = "\
Usage: tree_of_thought [OPTIONS]

A program to perform a 3-level tree of thought (Propose, Execute, Reflect) search.

Options:
  -q, --question <TEXT>        The question to solve [default: What is 12345 + 54321?]
  -b, --num-branches <INT>     Number of branches at each level of the tree [default: 2]
  -t, --max-tokens <INT>       Max new tokens to generate at each step [default: 128]
  -h, --help                   Prints this help message";

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
        .opt_value_from_str(["-q", "--question"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "What is the sum of 12345 and 54321?".to_string());
    let num_branches: usize = args
        .opt_value_from_str(["-b", "--num-branches"])
        .map_err(|e| e.to_string())?
        .unwrap_or(2);
    let max_tokens_per_step: usize = args
        .opt_value_from_str(["-t", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(128);

    let total_leaves = num_branches.pow(3);
    println!(
        "--- Starting Tree of Thought (Branches={}, Leaves={}, MaxTokens/Step={}) ---",
        num_branches, total_leaves, max_tokens_per_step
    );
    let start = Instant::now();

    // --- 2. Initialize Model and Root Context ---
    let model = inferlet::get_auto_model();
    let mut ctx_root = model.create_context();
    ctx_root.fill(SYSTEM_PROMPT);

    // --- 3. Build and Execute Tree in Parallel ---
    let level1_futures = (0..num_branches).map(|_| {
        let mut propose_ctx = ctx_root.fork();
        let question_ = question.clone();
        async move {
            // Level 1: Propose Plan
            let propose_prompt_full =
                format!("{}{}{}", USER_HEADER, PROPOSE_PROMPT_TEMPLATE, question_);
            propose_ctx.fill(&propose_prompt_full);
            propose_ctx.fill(EOT_ID);
            propose_ctx.fill(ASSISTANT_HEADER);
            propose_ctx
                .generate_until(max_tokens_per_step)
                .await;

            // Level 2: Execute Plan
            let level2_futures = (0..num_branches).map(|_| {
                let mut execute_ctx = propose_ctx.fork();
                async move {
                    execute_ctx.fill(USER_HEADER);
                    execute_ctx.fill(EXECUTE_PROMPT);
                    execute_ctx.fill(EOT_ID);
                    execute_ctx.fill(ASSISTANT_HEADER);
                    execute_ctx
                        .generate_until(max_tokens_per_step)
                        .await;

                    // Level 3: Reflect on Solution
                    let level3_futures = (0..num_branches).map(|_| {
                        let mut reflect_ctx = execute_ctx.fork();
                        async move {
                            reflect_ctx.fill(USER_HEADER);
                            reflect_ctx.fill(REFLECT_PROMPT);
                            reflect_ctx.fill(EOT_ID);
                            reflect_ctx.fill(ASSISTANT_HEADER);
                            reflect_ctx
                                .generate_until(max_tokens_per_step)
                                .await;
                            reflect_ctx // Return the final context for this leaf
                        }
                    });
                    join_all(level3_futures).await
                }
            });
            join_all(level2_futures).await
        }
    });

    // Await all top-level branches, which in turn await their sub-branches
    let nested_results: Vec<Vec<Vec<Context>>> = join_all(level1_futures).await;

    // Flatten the results to get a single list of all leaf contexts
    let final_ctxs: Vec<Context> = nested_results.into_iter().flatten().flatten().collect();

    // --- 4. Print Results ---
    println!(
        "\n--- All {} leaf nodes generated in {:?} ---\n",
        final_ctxs.len(),
        start.elapsed()
    );

    if let Some(last_ctx) = final_ctxs.last() {
        println!(
            "Sample Result (Leaf #{}):\n{}\n",
            final_ctxs.len(),
            last_ctx.get_text()
        );
    } else {
        println!("No results were generated.");
    }

    Ok(())
}
