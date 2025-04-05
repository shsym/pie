use std::future::Future;
use std::pin::Pin;
use symphony::{Context, Model};

// Constants
const DIVIDE_PROMPT_TEMPLATE: &str = "Your task is to analyze the given problem and decide whether it can be solved directly or needs to be divided into smaller subproblems. If the problem is simple and can be solved immediately, provide the solution wrapped in <leaf>final answer</leaf>. If not, divide the problem into exactly two independent subtasks such that solving these subtasks and combining their solutions will lead to the solution of the original problem. Present the subtasks wrapped in <branch>subtask1</branch> and <branch>subtask2</branch>. Be concise and ensure the subtasks are distinct and solvable. Problem: {}";
const SOLVE_PROMPT: &str =
    "Now, please solve the problem. Reason step-by-step. Make your response short.";
const MERGE_PROMPT: &str =
    "Now, please merge the two solutions into one. Make your response short.";
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";
const MAX_TOKENS: usize = 256;

// Type alias for a boxed future (single-threaded, no Send bound)
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;
// Parses the model's response to extract either a leaf answer or two branch subtasks.
fn parse_response(response: &str) -> Result<(Option<String>, Option<(String, String)>), String> {
    if let Some(start) = response.find("<leaf>") {
        if let Some(end) = response.find("</leaf>") {
            let answer = response[start + 6..end].trim().to_string();
            return Ok((Some(answer), None));
        }
    } else {
        let mut branches = Vec::new();
        let mut remaining = response;
        while let Some(start) = remaining.find("<branch>") {
            if let Some(end) = remaining.find("</branch>") {
                let task = remaining[start + 8..end].trim().to_string();
                branches.push(task);
                remaining = &remaining[end + 9..];
            } else {
                break;
            }
        }
        if branches.len() == 2 {
            return Ok((None, Some((branches[0].clone(), branches[1].clone()))));
        }
    }
    Err("Invalid response format: expected one <leaf> or two <branch> tags".to_string())
}

// Recursively divides a problem into subtasks, solves them, and merges the solutions.
fn divide_and_conquer(
    ctx: Context,
    question: &str, // Reference with lifetime 'a
    depth: usize,
    max_depth: usize,
) -> BoxFuture<String> {
    // Return a future tied to 'a
    Box::pin(async move {
        let mut ctx = ctx;
        if depth >= max_depth {
            let solve_prompt = format!("{} {}", SOLVE_PROMPT, question);
            ctx.fill(&solve_prompt);
            ctx.fill(ASSISTANT_PREFIX);
            return ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;
        }

        let divide_prompt = format!("{} {}", DIVIDE_PROMPT_TEMPLATE, question);
        ctx.fill(&divide_prompt);
        ctx.fill(ASSISTANT_PREFIX);
        let response = ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

        match parse_response(&response) {
            Ok((Some(answer), None)) => answer,
            Ok((None, Some((task1, task2)))) => {
                let solution1_future =
                    divide_and_conquer(ctx.fork(), task1.as_str(), depth + 1, max_depth);
                let solution2_future =
                    divide_and_conquer(ctx.fork(), task2.as_str(), depth + 1, max_depth);
                let (solution1, solution2) = futures::join!(solution1_future, solution2_future);

                let merge_prompt = format!(
                    "Subtask 1 solution: {}\nSubtask 2 solution: {}\n{}",
                    solution1, solution2, MERGE_PROMPT
                );
                let mut merge_ctx = ctx.fork();
                merge_ctx.fill(&merge_prompt);
                merge_ctx.fill(ASSISTANT_PREFIX);
                merge_ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await
            }
            Err(e) => format!("Error: {}", e),
            _ => "Error: Invalid response format".to_string(),
        }
    })
}
#[symphony::main]
async fn main() -> Result<(), String> {
    // Initialize the Symphony model and context
    let available_models = symphony::available_models();
    let model = Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    // Example question
    let question = "What is the sum of 123456789 and 987654321?";
    let max_depth = 5;

    // Execute Recursion-of-Thought and print the result
    let solution_future = divide_and_conquer(ctx, question, 0, max_depth);
    let solution = solution_future.await;
    println!("Final solution: {}", solution);

    Ok(())
}
