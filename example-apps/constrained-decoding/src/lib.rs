mod tokenizer;

use crate::tokenizer::BytePairEncoder;
use inferlet::sampler::Sampler;
use inferlet::traits::Tokenize;
use inferlet::traits::tokenize::Tokenizer;
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use pico_args::Arguments;
use std::collections::HashMap;
use std::ffi::OsString;
use std::time::Instant;

/// Default grammar to use if none is provided via command-line arguments.
const JSON_GRAMMAR: &str = r##"
?start: value
?value: object
        | array
        | string
        | SIGNED_NUMBER      -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null
array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" value
string : ESCAPED_STRING
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"##;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: program [OPTIONS]

A constrained sampler inferlet using a user-provided grammar.

Options:
  -p, --prompt <STRING>    The prompt to send to the model.
                           (default: "what is the capital of France? output data")
  -g, --grammar <STRING>   The Lark grammar to constrain the output.
                           (default: A JSON grammar)
  -n, --max-tokens <INT>   The maximum number of new tokens to generate.
                           (default: 128)
  -h, --help               Print help information.
"#;

struct ConstrainedSampler {
    pub constraint: Matcher, // Public to allow advancing tokens from main
    eos_token_id: u32,
}

impl ConstrainedSampler {
    pub fn new(tokenizer: Tokenizer, lark: &str, eos_token_id: u32) -> Self {
        let (ranks, words) = tokenizer.get_vocabs();
        let rank_map: HashMap<u32, Vec<u8>> = ranks.into_iter().zip(words).collect();

        let mut special_tokens = HashMap::new();

        special_tokens.insert("<|begin_of_text|>".to_string(), 128000);
        special_tokens.insert("<|end_of_text|>".to_string(), 128001);
        special_tokens.insert("<|start_header_id|>".to_string(), 128006);
        special_tokens.insert("<|end_header_id|>".to_string(), 128007);
        special_tokens.insert("<|eot_id|>".to_string(), 128009);

        let pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

        let tokenizer = BytePairEncoder::new(rank_map, special_tokens, pattern, eos_token_id);
        let tokenizer_env = tokenizer.unwrap().to_env();

        let grm = TopLevelGrammar::from_lark(lark.to_string());

        let factory = ParserFactory::new_simple(&tokenizer_env).unwrap();
        let parser = factory.create_parser(grm);

        let constraint = Matcher::new(parser);
        ConstrainedSampler {
            constraint,
            eos_token_id,
        }
    }
}

impl Sampler for ConstrainedSampler {
    fn sample(&mut self, token_ids: &[u32], probs: &[f32]) -> u32 {
        let res = self.constraint.compute_mask();

        if let Err(e) = res {
            return self.eos_token_id;
        }

        let res = res.unwrap();

        if res.is_empty() {
            return self.eos_token_id;
        }

        let mut max_prob = f32::NEG_INFINITY;
        let mut best_token = None;

        // Find the highest-probability token that is allowed by the grammar mask.
        for (i, &token_id) in token_ids.iter().enumerate() {
            if res.is_allowed(token_id) {
                if probs[i] > max_prob {
                    max_prob = probs[i];
                    best_token = Some(token_id);
                }
            }
        }

        let sampled_token_id = if let Some(token) = best_token {
            token
        } else {
            //println!("\n[Warning] No valid token found in model's candidates.");
            return res.first_bit_set().unwrap_or(0) as u32;
        };

        // Commit the chosen token to advance the parser's state.
        self.constraint.consume_token(sampled_token_id).unwrap();

        sampled_token_id
    }
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    // 1. Get arguments from the inferlet environment and prepare the parser.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
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
        .unwrap_or_else(|| "what is the capital of France? output data".to_string());

    let grammar = args
        .opt_value_from_str(["-g", "--grammar"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| JSON_GRAMMAR.to_string());

    let max_tokens: usize = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(128);

    // 4. Ensure no unknown arguments were passed.
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    // --- Main logic starts here ---
    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let mut ctx = model.create_context();

    let eot_token_id = tokenizer.tokenize("<|eot_id|>")[0];

    let mut sampler = ConstrainedSampler::new(tokenizer.clone(), &grammar, eot_token_id);

    // Assemble the full prompt that the model will process
    let full_prompt = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt
    );

    // 6. Fill the model's context with the prompt
    ctx.fill(&full_prompt);

    // 8. Set up the stop condition.
    let mut stop_condition = inferlet::stop_condition::any(
        inferlet::stop_condition::Until::new(vec![eot_token_id]),
        inferlet::stop_condition::Length::new(max_tokens),
    );

    // 9. Generate the output.
    let output = ctx.generate(&mut sampler, &mut stop_condition).await;
    let output_token_ids = tokenizer.tokenize(&output);

    println!(
        "\nOutput: {} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !output_token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (output_token_ids.len() as u32)
        );
    }

    Ok(())
}
