//! Demonstrates JSON Schema-validated generation with grammar-constrained decoding.
//!
//! Combines two layers:
//!  1. Deva's built-in `GrammarConstraint::from_json_schema` ensures every
//!     generated token keeps the JSON syntactically and structurally valid
//!     for the given schema.
//!  2. The `jsonschema` crate validates the decoded JSON; on validation
//!     failure the error messages are fed back to the model and it
//!     regenerates, still grammar-constrained.
//!
//! Deva's SDK has native JSON-schema → grammar conversion, so this example
//! is a thin wrapper — main's 400+ line custom ConstrainedSampler is not
//! needed here.

use inferlet::{
    Context, GrammarConstraint, Result, inference::Sampler, model::Model, runtime,
};
use serde_json::Value;

const HELP: &str = "\
Usage: json-schema-validation [OPTIONS]

Generates JSON from an LLM with grammar-constrained decoding, then
validates it against a JSON Schema. Retries with error feedback
until the output satisfies the schema.

Options:
  -p, --prompt <STRING>      The prompt describing what to generate
                             [default: a person profile]
  -r, --max-retries <N>      Maximum validation/retry cycles [default: 3]
  -t, --max-tokens <N>       Max tokens per generation attempt [default: 512]
  -h, --help                 Prints help information";

const PERSON_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        "email": {
            "type": "string"
        },
        "skills": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        },
        "address": {
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "country": { "type": "string" }
            },
            "required": ["city", "country"]
        }
    },
    "required": ["name", "age", "email", "skills", "address"]
}"#;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that generates structured data. When asked to produce \
JSON, you must output ONLY the raw JSON object with no additional text, markdown \
fences, or explanation. The JSON must be compact (no unnecessary whitespace). \
If you receive validation errors, fix exactly those issues and output the corrected \
JSON object.";

fn build_initial_prompt(user_prompt: &str, schema: &str) -> String {
    format!(
        "{}\n\nThe output must conform to this JSON Schema:\n{}\n\n\
         Output only the JSON object, nothing else.",
        user_prompt, schema
    )
}

fn build_retry_prompt(errors: &str) -> String {
    format!(
        "The JSON you produced has schema validation errors:\n{}\n\n\
         Please fix these errors and output only the corrected JSON object, nothing else.",
        errors
    )
}

/// Validate `json_text` against `schema_value`; return either the parsed
/// value or a human-readable error report aggregating all violations.
fn validate(json_text: &str, schema_value: &Value) -> std::result::Result<Value, String> {
    let parsed: Value = serde_json::from_str(json_text)
        .map_err(|e| format!("JSON parse error: {e}"))?;

    let validator = jsonschema::validator_for(schema_value)
        .map_err(|e| format!("Schema compile error: {e}"))?;

    let errors: Vec<String> = validator
        .iter_errors(&parsed)
        .map(|e| format!("- {} (at {})", e, e.instance_path()))
        .collect();

    if errors.is_empty() {
        Ok(parsed)
    } else {
        Err(errors.join("\n"))
    }
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let prompt: String = args.value_from_str(["-p", "--prompt"]).unwrap_or_else(|_| {
        "Generate a profile for a fictional software engineer named Alice.".to_string()
    });
    let max_retries: u32 = args.value_from_str(["-r", "--max-retries"]).unwrap_or(3);
    let max_tokens: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(512);

    let schema_value: Value =
        serde_json::from_str(PERSON_SCHEMA).map_err(|e| format!("Schema parse error: {e}"))?;

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&build_initial_prompt(&prompt, PERSON_SCHEMA));

    let mut valid_result: Option<Value> = None;

    for attempt in 1..=max_retries {
        println!("--- Attempt {}/{} ---", attempt, max_retries);

        // Fresh constraint per attempt so the matcher starts from the grammar's
        // start state. `cue()` re-inserts the assistant header before decoding.
        let constraint = GrammarConstraint::from_json_schema(PERSON_SCHEMA, &model)?;
        ctx.cue();

        let output = ctx
            .generate(Sampler::TopP((0.0, 1.0)))
            .with_max_tokens(max_tokens)
            .with_constraint(constraint)
            .collect_text()
            .await?;

        println!("Output: {}", output);

        match validate(&output, &schema_value) {
            Ok(parsed) => {
                println!("Schema validation passed!");
                valid_result = Some(parsed);
                break;
            }
            Err(error_report) => {
                println!("Validation errors:\n{}", error_report);
                // Treat the failed output as a prior assistant turn and ask
                // for a correction. The assistant message seal + new user
                // turn is handled by ctx.user below; the grammar-constrained
                // generation is reset by building a new GrammarConstraint
                // on the next loop iteration.
                ctx.assistant(&output);
                ctx.user(&build_retry_prompt(&error_report));
            }
        }
    }

    println!("\n--- Result ---");
    match valid_result {
        Some(result) => {
            println!(
                "Valid JSON:\n{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
            Ok(serde_json::to_string(&result).unwrap_or_default())
        }
        None => {
            println!("Failed to produce valid JSON after {} attempts.", max_retries);
            Err(format!("schema validation failed after {} attempts", max_retries))
        }
    }
}
