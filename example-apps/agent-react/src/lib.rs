use inferlet::wstd::time::Duration;
use pico_args::Arguments;
use rand::Rng;
use std::ffi::OsString;

// --- Help Message and Prompt Template ---
const HELP: &str = r#"
A benchmark script for ReAct-style function calling scenarios, implemented with inferlet.
This script is designed to be a direct comparison to its Python equivalent.

USAGE:
  benchmark [OPTIONS]

OPTIONS:
  -f, --num-function-calls <N>    Number of sequential function calls (Thought/Action/Observation cycles). [default: 3]
  -t, --tokens-between-calls <N>  Max tokens for the LLM to generate for each Thought/Action step. [default: 50]
  -d, --function-call-delay <MS>  Simulated delay in milliseconds for local tool/function execution. [default: 100]
  -h, --help                      Prints help information.
"#;

const REACT_PROMPT_TEMPLATE: &str = r#"
You are a helpful assistant that can use tools to answer questions. You have access to the following tools:

- `Search[query]`: Searches for information online.
- `Calculator[expression]`: Computes a mathematical expression.

To answer the user's question, you must break it down into a series of steps. For each step, you must first think about what to do, then output the action to take. The format should be:

Thought: Your reasoning for the next action.
Action: The tool to use, in the format `ToolName[input]`.

After you perform an action, you will receive an observation with the result. You will repeat this process until you have the final answer.

Question: Who is the director of the movie that won the Best Picture Oscar in the year the James Webb Space Telescope was launched?
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 1. Handle the --help flag.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // 2. Parse arguments, falling back to defaults that match the Python script.
    let num_function_calls: u32 = args
        .opt_value_from_str(["-f", "--num-function-calls"])
        .map_err(|e| e.to_string())?
        .unwrap_or(3);

    let tokens_between_calls: usize = args
        .opt_value_from_str(["-t", "--tokens-between-calls"])
        .map_err(|e| e.to_string())?
        .unwrap_or(50);

    let function_call_delay: u64 = args
        .opt_value_from_str(["-d", "--function-call-delay"])
        .map_err(|e| e.to_string())?
        .unwrap_or(100);

    // 3. Construct the initial user prompt with a random ID.
    let task_id = rand::rng().random_range(100_000..=999_999);
    let user_prompt = format!("TASK ID: {}\n{}", task_id, REACT_PROMPT_TEMPLATE);

    // 4. Set up the model and context.
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();

    // 5. Fill the context with the initial prompt structure.
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&user_prompt);

    // 6. Run the main agent loop, simulating function call cycles.
    for i in 0..num_function_calls {
        // Generate the Thought/Action from the LLM. The generated text is implicitly added to the context.
        ctx.generate_until(tokens_between_calls).await;

        // c. Simulate tool execution delay.
        if function_call_delay > 0 {
            inferlet::wstd::task::sleep(Duration::from_millis(function_call_delay)).await;
        }

        // d. Fill the context with a placeholder observation.
        ctx.fill(&format!("Observation: Result from function call {}", i + 1));
    }

    // The script's purpose is to benchmark the loop, so we finish here.
    Ok(())
}
