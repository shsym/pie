use futures::future::join_all;
use inferlet2::traits::Tokenize;
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: program [OPTIONS]

A simple inferlet to run a chat model.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           (default: "Explain the LLM decoding process ELI5.")
  -n, --max-tokens <INT>   The maximum number of new tokens to generate
                           (default: 256)
  --concurrent <INT>       Number of concurrent requests to run, splitting the prompt by ";"
                           (default: 1)
  --output                 Send the final output back to the user.
  -h, --help               Print help information
"#;

#[inferlet2::main]
async fn main() -> Result<(), String> {
    // 1. Get arguments from the inferlet environment and prepare the parser.
    let mut args = Arguments::from_vec(
        inferlet2::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 2. Handle the --help flag.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // 3. Parse arguments, falling back to defaults if they are not provided.
    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Explain the LLM decoding process ELI5.".to_string());

    let max_num_outputs: u32 = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(256);

    let concurrent: usize = args
        .opt_value_from_str("--concurrent")
        .map_err(|e| e.to_string())?
        .unwrap_or(1);

    // Check for the presence of the --output flag.
    let send_output = args.contains("--output");

    // Ensure no unknown arguments were passed.
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    // --- Main logic starts here ---
    let start = Instant::now();

    let model = inferlet2::get_auto_model();
    let tokenizer = model.get_tokenizer();

    // 4. Generate text based on whether execution is concurrent or single.
    let final_text = if concurrent > 1 {
        // Split prompt by ';' and filter out any empty strings.
        let prompts: Vec<_> = prompt.split(';').filter(|s| !s.trim().is_empty()).collect();

        // Create a future for each prompt.
        let futures = prompts.into_iter().map(|p| {
            let model = model.clone(); // Clone the model handle for each task.
            async move {
                let mut ctx = model.create_context();
                ctx.fill("<|begin_of_text|>");
                ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
                ctx.fill(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
                    p
                ));
                ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
                ctx.generate_until("<|eot_id|>", max_num_outputs as usize).await
            }
        });

        // Await all futures concurrently and collect the results.
        let results = join_all(futures).await;
        results.join("\n\n---\n\n") // Join the individual results into a single string.
    } else {
        // Fallback to the original single-prompt logic.
        let mut ctx = model.create_context();
        ctx.fill("<|begin_of_text|>");
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
        ctx.fill(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            prompt
        ));
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
        ctx.generate_until("<|eot_id|>", max_num_outputs as usize)
            .await
    };

    let token_ids = tokenizer.tokenize(&final_text);
    println!(
        "Output: {:?} (total elapsed: {:?})",
        final_text,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (token_ids.len() as u32)
        );
    }

    // Send back the output to the user only if the --output flag was provided.
    if send_output {
        inferlet2::send(&final_text);
    }

    Ok(())
}
