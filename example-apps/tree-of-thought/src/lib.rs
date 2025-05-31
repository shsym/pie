use futures::future::join_all;

use pie::{Context, Model};

const PROPOSE_PROMPT_TEMPLATE: &str = "Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: {}";
const EXECUTE_PROMPT: &str = "The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short.";
const REFLECT_PROMPT: &str = "Okay. Now you evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness.";
const ASSISTANT_PREFIX: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
const STOP_TOKEN: &str = "<|eot_id|>";
const MAX_TOKENS: usize = 32;

/// Asynchronously generates branches concurrently for proposing a plan.
async fn propose_plan(mut ctx: Context, question: &str, num_branches: usize) -> Vec<Context> {
    let prompt = format!("{} {}", PROPOSE_PROMPT_TEMPLATE, question);
    ctx.fill(&prompt);
    let branch_futures = (0..num_branches).map(|_| {
        let mut fork = ctx.fork();
        async move {
            fork.fill(ASSISTANT_PREFIX);
            fork.generate_until(STOP_TOKEN, MAX_TOKENS).await;
            fork
        }
    });
    let branches: Vec<Context> = join_all(branch_futures).await.into_iter().collect();
    branches
}

/// Asynchronously generates branches concurrently for executing a plan.
async fn execute_plan(mut ctx: Context, num_branches: usize) -> Vec<Context> {
    ctx.fill(EXECUTE_PROMPT);
    let branch_futures = (0..num_branches).map(|_| {
        let mut fork = ctx.fork();
        async move {
            fork.fill(ASSISTANT_PREFIX);
            fork.generate_until(STOP_TOKEN, MAX_TOKENS).await;
            fork
        }
    });
    let branches: Vec<Context> = join_all(branch_futures).await.into_iter().collect();
    branches
}

/// Asynchronously generates branches concurrently for reflecting on the solution.
async fn reflect_solution(mut ctx: Context, num_branches: usize) -> Vec<Context> {
    ctx.fill(REFLECT_PROMPT);
    let branch_futures = (0..num_branches).map(|_| {
        let mut fork = ctx.fork();
        async move {
            fork.fill(ASSISTANT_PREFIX);
            fork.generate_until(STOP_TOKEN, MAX_TOKENS).await;
            fork
        }
    });
    let branches: Vec<Context> = join_all(branch_futures).await.into_iter().collect();
    branches
}

async fn tree_search_branch_parallel(
    mut init_ctx: Context,
    question: &str,
    num_branches: usize,
) -> Vec<String> {
    // Define prompts as constants for clarity

    // --- Level 1: Propose Plan ---
    let propose_prompt = format!("{} {}", PROPOSE_PROMPT_TEMPLATE, question);
    init_ctx.fill(&propose_prompt);
    init_ctx.fill(ASSISTANT_PREFIX);

    let level1_futures = (0..num_branches).map(|_| {
        let mut propose_ctx = init_ctx.fork(); // Fork for the first level branch
        async move {
            propose_ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

            // --- Level 2: Execute Plan (nested within propose future) ---
            propose_ctx.fill(EXECUTE_PROMPT); // Add execute prompt to the *same* context
            propose_ctx.fill(ASSISTANT_PREFIX);

            let level2_futures = (0..num_branches).map(|_| {
                let mut execute_ctx = propose_ctx.fork(); // Fork from the propose context for the second level branch
                async move {
                    execute_ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

                    // --- Level 3: Reflect Solution (nested within execute future) ---
                    execute_ctx.fill(REFLECT_PROMPT); // Add reflect prompt to the *same* context
                    execute_ctx.fill(ASSISTANT_PREFIX);

                    let level3_futures = (0..num_branches).map(|_| {
                        let mut reflect_ctx = execute_ctx.fork(); // Fork from the execute context for the third level branch (leaf)
                        async move {
                            reflect_ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;
                            reflect_ctx // Return the final context for this leaf
                        }
                    });
                    // Await all reflection branches stemming from this execution branch
                    join_all(level3_futures).await
                }
            });
            // Await all execution branches stemming from this proposal branch, collecting Vec<Vec<Context>>
            join_all(level2_futures).await
        }
    });

    // Await all proposal branches, collecting Vec<Vec<Vec<Context>>>
    let nested_results: Vec<Vec<Vec<Context>>> = join_all(level1_futures).await;

    // Flatten the results to get a list of all leaf contexts
    let final_ctxs: Vec<Context> = nested_results
        .into_iter()
        .flatten() // Flattens Vec<Vec<Vec<Context>>> to Vec<Vec<Context>>
        .flatten() // Flattens Vec<Vec<Context>> to Vec<Context>
        .collect();

    // Collect the final output text from each leaf context.
    let outputs = final_ctxs.into_iter().map(|ctx| ctx.get_text()).collect();
    outputs
}

async fn tree_search_naive(
    mut init_ctx: Context,
    question: &str,
    num_branches: usize,
) -> Vec<String> {
    let propose_prompt = format!("{} {}", PROPOSE_PROMPT_TEMPLATE, question);

    init_ctx.fill(&propose_prompt);
    init_ctx.fill(ASSISTANT_PREFIX);

    let leaf_futures = (0..num_branches.pow(3))
        .map(|_| {
            let mut ctx = init_ctx.fork();
            async move {
                ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

                ctx.fill(EXECUTE_PROMPT);
                ctx.fill(ASSISTANT_PREFIX);
                ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

                ctx.fill(REFLECT_PROMPT);
                ctx.fill(ASSISTANT_PREFIX);
                ctx.generate_until(STOP_TOKEN, MAX_TOKENS).await;

                ctx.get_text()
            }
        })
        .collect::<Vec<_>>();

    join_all(leaf_futures).await
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

#[pie::main]
async fn main() -> Result<(), String> {
    // Initialize the Symphony model and a common context.
    let available_models = pie::available_models();

    let model = Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");

    let question = "What is the sum of 123456789 and 987654321?";
    let num_branches = 3;

    //tree_search(ctx, question, num_branches).await;
    //tree_search_branch_parallel(ctx, question, num_branches).await;\
    let res = tree_search_naive(ctx, question, num_branches).await;
    // Print the last result
    if let Some(last_result) = res.last() {
        println!("Final result: {}", last_result);
    } else {
        println!("No results generated.");
    }
    Ok(())
}
