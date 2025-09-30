use boa_engine::{Context, Source};
use pico_args::Arguments;
use std::ffi::OsString;

// --- Constants and Templates ---
const HELP: &str = r#"
A reproducible hybrid benchmark for CodeACT-style agentic workflows.
It calls a live LLM for realistic latency but uses hardcoded responses for deterministic logic.

USAGE:
  benchmark [OPTIONS]

OPTIONS:
  -f, --num-function-calls <N>    Number of sequential code execution cycles. [default: 2]
  -t, --tokens-between-calls <N>  Max tokens to ask the LLM to generate (to simulate workload). [default: 256]
  -h, --help                      Prints help information.
"#;

const CODEACT_PROMPT_TEMPLATE: &str = r#"
You are CodeACT, a highly intelligent AI assistant that can understand and execute JavaScript code to solve problems.
You will be given a task. To solve it, you must think step-by-step and produce JavaScript code in ```javascript ... ``` blocks.
You will receive the output of the code execution and repeat the process until you have the final answer.
"#;

// This hardcoded sequence simulates a perfect multi-turn code generation process.
// It is used INSTEAD of the LLM's actual output to ensure reproducibility.
const HARDCODED_RESPONSES: &[&str] = &[
    // Turn 1: Define the isPrime helper function.
    r#"Thought: I need to find the first 10 prime numbers and sum them. First, I need a way to determine if a number is prime. I will write a helper function `isPrime`.
```javascript
function isPrime(num) {
  if (num <= 1) return false;
  if (num <= 3) return true;
  if (num % 2 === 0 || num % 3 === 0) return false;
  for (let i = 5; i * i <= num; i = i + 6) {
    if (num % i === 0 || num % (i + 2) === 0) return false;
  }
  return true;
}
// The function is now defined for this execution context.
"isPrime function created";
```"#,
    // Turn 2: Use the function to find and sum the primes.
    r#"Thought: The `isPrime` function was defined successfully. Now I will use it to find the first 10 prime numbers and calculate their sum. I must include the `isPrime` definition again since the execution environment is stateless.
```javascript
function isPrime(num) {
  if (num <= 1) return false;
  if (num <= 3) return true;
  if (num % 2 === 0 || num % 3 === 0) return false;
  for (let i = 5; i * i <= num; i = i + 6) {
    if (num % i === 0 || num % (i + 2) === 0) return false;
  }
  return true;
}

let primes = [];
let num = 2;
while (primes.length < 10) {
  if (isPrime(num)) {
    primes.push(num);
  }
  num++;
}
primes.reduce((a, b) => a + b, 0);
```"#,
    // Turn 3: Final answer (no code).
    r#"Thought: I have successfully calculated the sum of the first 10 prime numbers. The result from the code execution was 129. I will now state the final answer clearly.

The final answer is that the sum of the first 10 prime numbers is 129."#,
];

// --- Main Application Logic ---
#[inferlet::main]
async fn main() -> Result<(), String> {
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

    // Parse arguments
    let num_function_calls: usize = args
        .opt_value_from_str(["-f", "--num-function-calls"])
        .map_err(|e| e.to_string())?
        .unwrap_or(2);
    let tokens_between_calls: usize = args
        .opt_value_from_str(["-t", "--tokens-between-calls"])
        .map_err(|e| e.to_string())?
        .unwrap_or(256);

    // Set up the model and context
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();
    ctx.fill_system(CODEACT_PROMPT_TEMPLATE);
    ctx.fill_user("Calculate the sum of the first 10 prime numbers.");

    // --- Agentic Loop ---
    let num_code_turns = std::cmp::min(num_function_calls, HARDCODED_RESPONSES.len() - 1);

    for i in 0..num_code_turns {
        // 1. Call the real LLM to include its inference delay in the benchmark.
        //    The actual response is discarded to ensure deterministic behavior.
        let _ = ctx.generate_until(tokens_between_calls).await;

        // 2. Use the hardcoded response for this turn to control the logic.
        let assistant_response = HARDCODED_RESPONSES[i];

        // 4. Extract and execute JS code from the hardcoded response.
        if let Some(js_code) = extract_js_code(assistant_response) {
            let result = execute_js_code(&js_code);
            ctx.fill(&format!("<|start_header_id|>system<|end_header_id|>\n\nCode execution result:\nOutput: {}<|eot_id|>", result));
        } else {
            ctx.fill(
                "<|start_header_id|>system<|end_header_id|>\n\nNo code was executed.<|eot_id|>",
            );
        }

        // 5. Prepare for the next turn.
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    // 6. Perform a final LLM call and use the final hardcoded text response.
    let _ = ctx.generate_until(tokens_between_calls).await;

    Ok(())
}

// --- Helper Functions ---

/// Extracts JavaScript code from a markdown block.
fn extract_js_code(text: &str) -> Option<String> {
    let start_marker = "```javascript";
    let end_marker = "```";
    if let Some(start) = text.find(start_marker) {
        let code_start = start + start_marker.len();
        if let Some(end) = text[code_start..].find(end_marker) {
            let code = &text[code_start..code_start + end];
            return Some(code.trim().to_string());
        }
    }
    None
}

/// Executes the given JavaScript code using the Boa engine.
fn execute_js_code(code: &str) -> String {
    let mut context = Context::default();
    match context.eval(Source::from_bytes(code)) {
        Ok(res) => res
            .to_string(&mut context)
            .unwrap_or_else(|_| "undefined".into())
            .to_std_string()
            .unwrap(),
        Err(e) => format!("Execution Error: {}", e),
    }
}
