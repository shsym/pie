use futures::future::join_all;
use inferlet::Context;
use inferlet::traits::Allocate;
use inferlet::wstd::time::Duration;
use pico_args::Arguments;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;

// --- Help Message and Prompt Template ---
const HELP: &str = r#"
A benchmark script for optimized ReAct-style function calling scenarios.

USAGE:
  benchmark [OPTIONS]

OPTIONS:
  -t, --tokens-between-calls <N>  Max tokens for the LLM to generate for each Thought/Action step. [default: 50]
  -d, --function-call-delay <MS>  Simulated delay in milliseconds for local tool/function execution. [default: 100]
      --use-prefix-cache          Enable caching of the initial system prompt KV pages between runs.
      --drop-tool-cache           Enable dropping of the first tool's KV cache after it's no longer needed.
      --concurrent-calls          Simulate concurrent execution of function calls.
  -h, --help                      Prints help information.
"#;

const TEMPLATE_INTRO: &str = r#"
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant that can use tools to answer questions. You have access to the following tools:
"#;

const WEB_SEARCH_DOCS: &str = r#"
- `WebSearch[query]`: This tool allows you to perform powerful, semantic searches across a vast corpus of indexed web pages, academic papers, and news articles. It is ideal for finding specific facts, dates, names, definitions, and general knowledge. The query should be a concise question or a set of keywords. For example, to find the capital of France, you would use `WebSearch[capital of France]`. The tool returns a snippet of the most relevant document. It is particularly effective for questions about recent events or information that might change over time. The search index is updated continuously, ensuring access to the most current information available. It can handle complex, natural language queries and is optimized for fact-finding missions. Avoid overly broad queries; be as specific as possible to get the best results. The underlying engine uses a combination of keyword matching and vector-based similarity search to identify the most relevant information, making it robust against variations in phrasing and terminology. This is your primary tool for gathering new information that is not already in your knowledge base. It is not suitable for mathematical calculations or for performing actions in the real world. It only retrieves information.
"#;

const CODE_INTERPRETER_DOCS: &str = r#"
- `CodeInterpreter[code]`: This tool executes Python code in a sandboxed environment. It is perfect for complex calculations, data analysis, simulations, and any task that can be expressed as a program. The input `code` must be a valid Python script. The final line of the script should be an expression or a `print()` statement that produces the output. For example, `CodeInterpreter[print(sum([i*i for i in range(100)]))]`. You can use popular libraries like `pandas`, `numpy`, and `matplotlib`. The environment is stateful within a single turn, but the state is reset for each new call. This tool is essential for tasks requiring precise numerical computation, data manipulation, or algorithmic logic. Do not use it for simple arithmetic that can be done manually. The sandbox has no internet access; any required data must be passed directly into the `code` parameter. It is a powerful tool for any problem that requires logic, iteration, or complex mathematical formulas. Ensure the code is self-contained and produces a clear output, as this output will be the only thing returned to you as an observation. This is your go-to for quantitative reasoning.
"#;

const TEMPLATE_OUTRO: &str = r#"
To answer the user's question, you must break it down into a series of steps. For each step, you must first think about what to do, then output the action to take. The format should be:

Thought: Your reasoning for the next action.
Action: The tool to use, in the format `ToolName[input]`.

After you perform an action, you will receive an observation with the result. You will repeat this process until you have the final answer.
<|eot_id|>
"#;

// --- Cache and State Management ---
const CACHE_FLAG_KEY: &str = "prefix_loaded_v6";
const CACHE_EXPORT_NAME: &str = "my_system_prefix_v6";
const CACHE_STATE_KEY: &str = "my_system_prefix_state_v6";

#[derive(Serialize, Deserialize)]
struct CachedPrefixState {
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

#[derive(Clone)]
struct ToolTokenRange {
    start: usize,
    end: usize,
}

// --- Predefined ReAct Sequences ---
const PREDEFINED_SEQUENCES: [(&str, &str); 6] = [
    // 3x Independent WebSearch calls
    (
        "\nThought: The user is asking a factual question about geography. I will use WebSearch to find the highest mountain in North America.\nAction: WebSearch[highest mountain in North America]",
        "\nObservation: Denali (formerly known as Mount McKinley) is the highest mountain peak in North America, with a summit elevation of 20,310 feet (6,190 m) above sea level.",
    ),
    (
        "\nThought: The user is asking a question about classic literature. I will use WebSearch to find the author of 'Pride and Prejudice'.\nAction: WebSearch[author of Pride and Prejudice]",
        "\nObservation: 'Pride and Prejudice' is a romantic novel of manners written by Jane Austen, published in 1813.",
    ),
    (
        "\nThought: The user wants to know a specific chemical formula. WebSearch is the appropriate tool for this.\nAction: WebSearch[chemical formula for caffeine]",
        "\nObservation: The chemical formula for caffeine is C8H10N4O2.",
    ),
    // 3x CodeInterpreter calls
    (
        "\nThought: Now I need to perform a calculation. I will calculate the sum of squares from 1 to 50.\nAction: CodeInterpreter[print(sum([i**2 for i in range(1, 51)]))]",
        "\nObservation: 42925",
    ),
    (
        "\nThought: I will perform another calculation. I'll find the 20th Fibonacci number.\nAction: CodeInterpreter[a, b = 0, 1\nfor _ in range(19):\n  a, b = b, a + b\nprint(a)]",
        "\nObservation: 6765",
    ),
    (
        "\nThought: One final calculation. I'll approximate pi using the Nilakantha series with 1000 terms.\nAction: CodeInterpreter[pi = 3.0\nsign = 1\nfor i in range(2, 2001, 2):\n  pi += sign * 4 / (i * (i + 1) * (i + 2))\n  sign *= -1\nprint(pi)]",
        "\nObservation: 3.1415921535897914",
    ),
];

// Helper to find the token indices of tool docs within the full system prompt.
fn get_tool_token_ranges(model: &inferlet::Model) -> ToolTokenRange {
    let intro_tokens = model.get_tokenizer().tokenize(TEMPLATE_INTRO).len();
    let search_tokens = model.get_tokenizer().tokenize(WEB_SEARCH_DOCS).len();

    let search_start = intro_tokens;
    let search_end = search_start + search_tokens;

    ToolTokenRange {
        start: search_start,
        end: search_end,
    }
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 1. Handle command-line arguments
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let tokens_between_calls: usize = args
        .opt_value_from_str(["-t", "--tokens-between-calls"])
        .map_err(|e| e.to_string())?
        .unwrap_or(50);

    let function_call_delay_ms: u64 = args
        .opt_value_from_str(["-d", "--function-call-delay"])
        .map_err(|e| e.to_string())?
        .unwrap_or(100);
    let function_call_delay = Duration::from_millis(function_call_delay_ms);

    let use_prefix_cache = args.contains("--use-prefix-cache");
    let drop_tool_cache = args.contains("--drop-tool-cache");
    let concurrent_calls = args.contains("--concurrent-calls");

    println!("\n--- Benchmark Configuration ---");
    println!(
        "Prefix Caching: {}",
        if use_prefix_cache { "ON" } else { "OFF" }
    );
    println!(
        "Tool Cache Dropping: {}",
        if drop_tool_cache { "ON" } else { "OFF" }
    );
    println!(
        "Concurrent Calls: {}",
        if concurrent_calls {
            "Concurrent"
        } else {
            "Sequential"
        }
    );
    println!("Tokens per Generation: {}", tokens_between_calls);
    println!("-----------------------------\n");

    // 2. Setup model and context based on optimizations
    let full_system_prompt = format!(
        "{}{}{}{}",
        TEMPLATE_INTRO, WEB_SEARCH_DOCS, CODE_INTERPRETER_DOCS, TEMPLATE_OUTRO
    );
    let model = inferlet::get_auto_model();
    let web_search_range = get_tool_token_ranges(&model);
    let mut ctx = model.create_context();

    if use_prefix_cache {
        let queue = ctx.queue();
        if inferlet::store_get(CACHE_FLAG_KEY) == Some("true".to_string()) {
            println!("Cache HIT. Loading prefix from KV store.");
            let imported_page_ids = queue.import_kv_pages(CACHE_EXPORT_NAME.to_string());
            let state_json = inferlet::store_get(CACHE_STATE_KEY)
                .ok_or("Cache Inconsistency: State missing")?;
            let state: CachedPrefixState = serde_json::from_str(&state_json).unwrap();
            ctx = Context::from_imported_state(
                &model,
                imported_page_ids,
                state.token_ids,
                state.kv_page_last_len,
            );
        } else {
            println!("Cache MISS. Computing and caching system prompt prefix.");
            let mut prefill_ctx = model.create_context();
            prefill_ctx.fill(&full_system_prompt);
            prefill_ctx.flush();

            let page_ids = prefill_ctx.get_kv_page_ids().to_vec();
            let state_to_cache = CachedPrefixState {
                token_ids: prefill_ctx.get_token_ids().to_vec(),
                kv_page_last_len: prefill_ctx.get_kv_page_last_len(),
            };

            prefill_ctx
                .queue()
                .export_kv_pages(&page_ids, CACHE_EXPORT_NAME.to_string(), true);

            let state_json = serde_json::to_string(&state_to_cache).unwrap();
            inferlet::store_set(CACHE_STATE_KEY, &state_json);
            inferlet::store_set(CACHE_FLAG_KEY, "true");

            ctx = prefill_ctx;
        }
    } else {
        println!("Prefix caching disabled. Filling prompt directly.");
        ctx.fill(&full_system_prompt);
    }

    // 3. Construct initial user prompt and fill context
    let user_prompt = "Perform a series of research and calculation tasks.";

    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{}",
        user_prompt
    ));
    ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");

    // 4. Main ReAct simulation
    if concurrent_calls {
        // --- CONCURRENT SIMULATION ---
        println!("\n--- Batch 1: WebSearch (Concurrent Simulation) ---");
        let mut futures = Vec::new();
        for i in 0..3 {
            println!("Generating Thought/Action #{}...", i + 1);
            let _ = ctx.generate_until("<|eot_id|>", tokens_between_calls).await;
            println!(
                "Simulating sequential call (delay: {}ms)...",
                function_call_delay_ms
            );
            futures.push(inferlet::wstd::task::sleep(function_call_delay));
        }

        println!("\nWaiting for 3 concurrent calls to complete...");
        join_all(futures).await;
        println!("All calls completed.");

        println!("\nReceived all observations. Adding to context...");
        for i in 0..3 {
            ctx.fill("<|eot_id|><|start_header_id|>tool<|end_header_id|>");
            ctx.fill(PREDEFINED_SEQUENCES[i].1);
        }
        ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");

        // --- OPTIONAL CACHE DROP ---
        if drop_tool_cache {
            println!("\n OPTIMIZATION: Finished with WebSearch tool.");
            let mut start_idx = web_search_range.start;
            if start_idx % 16 != 0 {
                start_idx += 16 - (start_idx % 16);
            }
            let mut end_idx = web_search_range.end;
            if end_idx % 16 != 0 {
                end_idx -= end_idx % 16;
            }

            println!(
                "Dropping KV cache for WebSearch docs (aligned tokens {}-{}).",
                start_idx, end_idx
            );
            ctx.mask_token_range(start_idx, end_idx, true);
            ctx.drop_masked_kv_pages();
        }

        // --- BATCH 2: CodeInterpreter ---
        println!("\n--- Batch 2: CodeInterpreter (Concurrent Simulation) ---");
        let mut futures = Vec::new();
        for i in 3..6 {
            println!("Generating Thought/Action #{}...", i + 1);
            ctx.generate_until("<|eot_id|>", tokens_between_calls).await;
            ctx.fill(PREDEFINED_SEQUENCES[i].0);
            println!(
                "Simulating sequential call (delay: {}ms)...",
                function_call_delay_ms
            );
            futures.push(inferlet::wstd::task::sleep(function_call_delay));
        }

        println!("\nWaiting for 3 concurrent calls to complete...");
        join_all(futures).await;
        println!("All calls completed.");

        println!("\nReceived all observations. Adding to context...");
        for i in 3..6 {
            ctx.fill("<|eot_id|><|start_header_id|>tool<|end_header_id|>");
            ctx.fill(PREDEFINED_SEQUENCES[i].1);
        }
        ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");
    } else {
        // --- SEQUENTIAL SIMULATION (UNROLLED LOOP) ---
        for i in 0..6 {
            let tool_name = if i < 3 {
                "WebSearch"
            } else {
                "CodeInterpreter"
            };
            println!("\n--- Turn {}: {} ---", i + 1, tool_name);
            println!("Generating Thought/Action...");
            ctx.generate_until("<|eot_id|>", tokens_between_calls).await;
            // For determinism, we fill the predefined action after generating.
            ctx.fill(PREDEFINED_SEQUENCES[i].0);

            println!(
                "Simulating sequential call (delay: {}ms)...",
                function_call_delay_ms
            );
            inferlet::wstd::task::sleep(function_call_delay).await;

            ctx.fill("<|eot_id|><|start_header_id|>tool<|end_header_id|>");
            ctx.fill(PREDEFINED_SEQUENCES[i].1);
            ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");

            // Optional cache drop after turn 3
            if i == 2 && drop_tool_cache {
                println!("\nOPTIMIZATION: Finished with WebSearch tool.");
                let mut start_idx = web_search_range.start;
                if start_idx % 16 != 0 {
                    start_idx += 16 - (start_idx % 16);
                }
                let mut end_idx = web_search_range.end;
                if end_idx % 16 != 0 {
                    end_idx -= end_idx % 16;
                }

                println!(
                    "Dropping KV cache for WebSearch docs (aligned tokens {}-{}).",
                    start_idx, end_idx
                );
                ctx.mask_token_range(start_idx, end_idx, true);
                ctx.drop_masked_kv_pages();
            }
        }
    }

    println!("\n✅ Benchmark complete.");

    Ok(())
}
