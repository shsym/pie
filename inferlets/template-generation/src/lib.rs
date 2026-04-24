//! Demonstrates template-driven generation with grammar-constrained decoding.
//!
//! Three layers:
//!  1. Deva's built-in `GrammarConstraint::from_json_schema` constrains every
//!     generated token to keep the output valid against the given JSON schema.
//!  2. `jsonschema` validates the decoded JSON against the schema.
//!  3. `minijinja` renders the validated data through a Jinja2-style template.
//!
//! On validation or render failure, the errors are fed back to the model and
//! it regenerates (still grammar-constrained). Retries cap at `--max-retries`.

use inferlet::{
    Context, GrammarConstraint, Result, inference::Sampler, model::Model, runtime,
};
use minijinja::Environment;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_prompt() -> String { "an AI-powered code editor".to_string() }
fn default_max_retries() -> u32 { 3 }
fn default_max_tokens() -> usize { 1024 }

const TEMPLATE: &str = r#"
========================================
  PRODUCT ANNOUNCEMENT
========================================

{{ product_name | upper }}
"{{ tagline }}"

OVERVIEW
--------
{{ description }}

KEY FEATURES
------------
{% for feature in features %}
  * {{ feature }}
{% endfor %}

PRICING & AVAILABILITY
----------------------
  Price: ${{ price }}
  Release Date: {{ release_date }}
{% if discount_percent %}
  Launch Discount: {{ discount_percent }}% off!
{% endif %}

========================================
"#;

const PRODUCT_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "product_name": {
            "type": "string",
            "minLength": 1
        },
        "tagline": {
            "type": "string",
            "minLength": 1
        },
        "description": {
            "type": "string",
            "minLength": 1
        },
        "features": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        },
        "price": {
            "type": "string"
        },
        "release_date": {
            "type": "string"
        },
        "discount_percent": {
            "type": ["integer", "null"]
        }
    },
    "required": ["product_name", "tagline", "description", "features", "price", "release_date"]
}"#;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant that generates structured product data. \
Output ONLY a raw JSON object with no additional text, markdown fences, or explanation. \
The JSON must conform to the JSON Schema provided in the user message. \
If you receive validation or rendering errors, fix the JSON to address the issues \
and output only the corrected JSON object.";

fn build_initial_prompt(user_prompt: &str, schema: &str) -> String {
    format!(
        "Generate product announcement data for: {}.\n\n\
         The output must conform to this JSON Schema:\n{}\n\n\
         Output only the JSON object, nothing else.",
        user_prompt, schema
    )
}

fn build_retry_prompt(errors: &str) -> String {
    format!(
        "The JSON you produced has validation/rendering errors:\n{}\n\n\
         Please fix these errors and output only the corrected JSON object, nothing else.",
        errors
    )
}

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

fn render(data: &Value) -> std::result::Result<String, String> {
    let mut env = Environment::new();
    env.add_template("announcement", TEMPLATE)
        .map_err(|e| format!("Template compile error: {e}"))?;
    let tmpl = env
        .get_template("announcement")
        .map_err(|e| format!("Template lookup error: {e}"))?;
    // Render with the validated JSON as the top-level context so fields
    // like product_name/tagline/features are directly accessible.
    tmpl.render(data)
        .map_err(|e| format!("Template render error: {e}"))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let prompt = input.prompt;
    let max_retries = input.max_retries;
    let max_tokens = input.max_tokens;

    let schema_value: Value =
        serde_json::from_str(PRODUCT_SCHEMA).map_err(|e| format!("Schema parse error: {e}"))?;

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&build_initial_prompt(&prompt, PRODUCT_SCHEMA));

    let mut rendered: Option<String> = None;

    for attempt in 1..=max_retries {
        println!("--- Attempt {}/{} ---", attempt, max_retries);

        let constraint = GrammarConstraint::from_json_schema(PRODUCT_SCHEMA, &model)?;
        ctx.cue();

        let output = ctx
            .generate(Sampler::TopP((0.0, 1.0)))
            .with_max_tokens(max_tokens)
            .with_constraint(constraint)
            .collect_text()
            .await?;

        println!("Output: {}", output);

        let validated = match validate(&output, &schema_value) {
            Ok(parsed) => parsed,
            Err(err) => {
                println!("Validation errors:\n{}", err);
                ctx.assistant(&output);
                ctx.user(&build_retry_prompt(&err));
                continue;
            }
        };

        match render(&validated) {
            Ok(text) => {
                println!("Rendered successfully.");
                rendered = Some(text);
                break;
            }
            Err(err) => {
                println!("Render error:\n{}", err);
                ctx.assistant(&output);
                ctx.user(&build_retry_prompt(&err));
            }
        }
    }

    println!("\n--- Result ---");
    match rendered {
        Some(text) => {
            println!("{}", text);
            Ok(text)
        }
        None => {
            println!("Failed to produce a valid rendered document after {} attempts.", max_retries);
            Err(format!(
                "template generation failed after {} attempts",
                max_retries
            ))
        }
    }
}
