use futures::future::join_all;
use regex::Regex;
use std::collections::HashMap;
use symphony::{Context, Model};

const INVALID: i64 = -9999999;
const TEMP: f32 = 0.3; // Temperature value if supported

// Extracts the last number from the answer string or returns INVALID.
fn get_answer_value(answer: &str) -> i64 {
    let cleaned = answer.replace(",", "");
    let re = Regex::new(r"\d+").unwrap();
    let numbers: Vec<&str> = re.find_iter(&cleaned).map(|m| m.as_str()).collect();
    if numbers.is_empty() {
        return INVALID;
    }
    numbers
        .last()
        .and_then(|num_str| num_str.parse::<i64>().ok())
        .unwrap_or(INVALID)
}

// Returns the most frequent number in the slice.
fn most_frequent_number(numbers: &[i64]) -> Option<i64> {
    if numbers.is_empty() {
        return None;
    }
    let mut freq = HashMap::new();
    for &num in numbers {
        *freq.entry(num).or_insert(0) += 1;
    }
    freq.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(num, _)| num)
}

/// Asynchronously generates branches concurrently for proposing a plan.
async fn propose_plan(mut ctx: Context, question: &str, num_branches: usize) -> Vec<Context> {
    let prompt = format!(
        "Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: {}",
        question
    );
    ctx.fill(&prompt);
    let branch_futures = (0..num_branches).map(|_| {
        let mut fork = ctx.fork();
        async move {
            fork.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
            fork.generate_until("<|eot_id|>", 256).await;
            fork
        }
    });
    let branches: Vec<Context> = join_all(branch_futures).await.into_iter().collect();
    branches
}

/// Asynchronously generates branches concurrently for executing a plan.
async fn execute_plan(mut ctx: Context, num_branches: usize) -> Vec<Context> {
    let prompt = "The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short.";
    ctx.fill(prompt);
    let branch_futures = (0..num_branches).map(|_| {
        let mut fork = ctx.fork();
        async move {
            fork.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
            fork.generate_until("<|eot_id|>", 256).await;
            fork
        }
    });
    let branches: Vec<Context> = join_all(branch_futures).await.into_iter().collect();
    branches
}

/// Asynchronously generates branches concurrently for reflecting on the solution.
async fn reflect_solution(mut ctx: Context, num_branches: usize) -> Vec<Context> {
    let prompt = "Okay. Now you evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness.";
    ctx.fill(prompt);
    let branch_futures = (0..num_branches).map(|_| {
        let mut fork = ctx.fork();
        async move {
            fork.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
            fork.generate_until("<|eot_id|>", 256).await;
            fork
        }
    });
    let branches: Vec<Context> = join_all(branch_futures).await.into_iter().collect();
    branches
}

/// Implements the tree search: propose a plan, execute it, then reflect on the solution.
async fn tree_search(init_ctx: Context, question: &str, num_branches: usize) -> Vec<String> {
    let plan_ctxs = propose_plan(init_ctx, question, num_branches).await;

    // Execute plan concurrently for each plan branch.
    let exec_futures = plan_ctxs
        .into_iter()
        .map(|plan_ctx| execute_plan(plan_ctx, num_branches));

    let exec_results = join_all(exec_futures).await;
    let mut solution_ctxs = Vec::new();
    for branches in exec_results {
        solution_ctxs.extend(branches);
    }

    // Reflect on the solutions concurrently.
    let reflect_futures = solution_ctxs
        .into_iter()
        .map(|sol_ctx| reflect_solution(sol_ctx, num_branches));
    let reflect_results = join_all(reflect_futures).await;
    let mut final_ctxs = Vec::new();
    for branches in reflect_results {
        final_ctxs.extend(branches);
    }

    // Collect the final output text from each context.
    let outputs = final_ctxs.into_iter().map(|ctx| ctx.get_text()).collect();
    outputs
}

#[symphony::main]
async fn main() -> Result<(), String> {
    // Initialize the Symphony model and a common context.
    let available_models = symphony::available_models();

    let model = Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    let question = "What is the sum of 123456789 and 987654321?";
    let num_branches = 3;

    tree_search(ctx, question, num_branches).await;

    Ok(())
}
