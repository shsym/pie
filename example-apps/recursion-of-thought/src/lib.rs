use futures::future;
use inferlet::{Context, Model};
use pico_args::Arguments;
use std::ffi::OsString;
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

// --- Constants ---
const HELP: &str = "\
Usage: rot [OPTIONS]

A program to solve a problem using a recursive divide-and-conquer strategy (Recursion-of-Thought).

Options:
  -q, --question <TEXT>   The initial question to solve [default: \"What is the sum of 123456789 and 987654321?\"]
  -d, --max-depth <NUM>   The maximum recursion depth [default: 5]
  -t, --max-tokens <NUM>  The max tokens to generate at each step [default: 128]
  -h, --help              Prints this help message";

const DIVIDE_PROMPT_TEMPLATE: &str = "Your task is to analyze the given problem and decide whether it can be solved directly or needs to be divided into smaller subproblems. If the problem is simple and can be solved immediately, provide the solution wrapped in <leaf>final answer</leaf>. If not, divide the problem into exactly two independent subtasks such that solving these subtasks and combining their solutions will lead to the solution of the original problem. Present the subtasks wrapped in <branch>subtask1</branch> and <branch>subtask2</branch>. Be concise and ensure the subtasks are distinct and solvable. Problem: {}";
const SOLVE_PROMPT: &str =
    "Now, please solve the problem. Reason step-by-step. Make your response short.";
const MERGE_PROMPT: &str =
    "Now, please merge the two solutions into one. Make your response short.";
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";

// --- Type Alias ---
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

// --- Core Logic ---

/// Parses the model's response to extract either a leaf answer or two branch subtasks.
fn parse_response(response: &str) -> Result<(Option<String>, Option<(String, String)>), String> {
    if let Some(start) = response.find("<leaf>") {
        if let Some(end) = response.find("</leaf>") {
            let answer = response[start + 6..end].trim().to_string();
            return Ok((Some(answer), None));
        }
    }

    let branches: Vec<String> = response
        .match_indices("<branch>")
        .zip(response.match_indices("</branch>"))
        .map(|((start, _), (end, _))| response[start + 8..end].trim().to_string())
        .collect();

    if branches.len() == 2 {
        Ok((None, Some((branches[0].clone(), branches[1].clone()))))
    } else {
        Err(format!(
            "Error: Expected a <leaf> tag or exactly two <branch> tags, but found {} branches.",
            branches.len()
        ))
    }
}

/// Recursively divides a problem, solves sub-problems, and merges the solutions.
fn divide_and_conquer<'a>(
    ctx: Context,
    question: &'a str,
    depth: usize,
    max_depth: usize,
    max_tokens: usize,
) -> BoxFuture<'a, String> {
    Box::pin(async move {
        // Base Case: If max depth is reached, solve the problem directly.
        if depth >= max_depth {
            let mut solve_ctx = ctx;
            let solve_prompt = format!("{} {}", SOLVE_PROMPT, question);
            solve_ctx.fill(&solve_prompt);
            solve_ctx.fill(ASSISTANT_PREFIX);
            return solve_ctx.generate_until(max_tokens).await;
        }

        // Recursive Step: Try to divide the problem.
        let mut divide_ctx = ctx.fork();
        let divide_prompt = format!("{}", DIVIDE_PROMPT_TEMPLATE.replace("{}", question));
        divide_ctx.fill(&divide_prompt);
        divide_ctx.fill(ASSISTANT_PREFIX);
        let response = divide_ctx.generate_until(max_tokens).await;

        match parse_response(&response) {
            // Case 1: The model provided a direct answer (leaf node).
            Ok((Some(answer), None)) => answer,
            // Case 2: The model divided the problem into two subtasks (branch node).
            Ok((None, Some((task1, task2)))) => {
                let solution1_future =
                    divide_and_conquer(ctx.fork(), &task1, depth + 1, max_depth, max_tokens);
                let solution2_future =
                    divide_and_conquer(ctx.fork(), &task2, depth + 1, max_depth, max_tokens);
                let (solution1, solution2) = future::join(solution1_future, solution2_future).await;

                let mut merge_ctx = ctx;
                let merge_prompt = format!(
                    "Subtask 1 solution: {}\nSubtask 2 solution: {}\n{}",
                    solution1, solution2, MERGE_PROMPT
                );
                merge_ctx.fill(&merge_prompt);
                merge_ctx.fill(ASSISTANT_PREFIX);
                merge_ctx.generate_until(max_tokens).await
            }
            // Case 3: Error in parsing the response.
            Err(e) => format!("Parsing Error: {}", e),
            // Case 4: Invalid response format.
            _ => "Error: Invalid response format from model.".to_string(),
        }
    })
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
        .opt_value_from_str(["-q", "--question"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "What is the sum of 123456789 and 987654321?".to_string());

    let max_depth: usize = args
        .opt_value_from_str(["-d", "--max-depth"])
        .map_err(|e| e.to_string())?
        .unwrap_or(5);

    let max_tokens: usize = args
        .opt_value_from_str(["-t", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(128);

    // --- 2. Initialization ---
    let start_time = Instant::now();
    println!("--- Initializing Model and Context ---");
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    // --- 3. Execution ---
    println!("--- Starting Recursion-of-Thought (RoT) ---");
    println!("Question: {}", question);
    println!("Max Depth: {}, Max Tokens: {}", max_depth, max_tokens);

    let solution = divide_and_conquer(ctx, &question, 0, max_depth, max_tokens).await;

    // --- 4. Print Results ---
    println!("\n--- ✅ RoT Complete in {:?} ---", start_time.elapsed());
    println!("\nFinal solution:\n{}", solution);

    Ok(())
}
