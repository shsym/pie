use std::time::Instant;
use symphony::RunSync;

struct ConstrainedDecoding;

fn llama3_format(prompt: &str, hint: Option<&str>, system: Option<&str>) -> String {
    let system_msg = system.unwrap_or("You are a helpful, respectful and honest assistant.");
    format!(
        "<|begin_of_text|>\
<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n{}",
        system_msg,
        prompt,
        hint.unwrap_or("")
    )
}

// create a default stream constant
const MAIN: u32 = 0;
const MAX_NUM_OUTPUTS: usize = 128;

impl RunSync for ConstrainedDecoding {
    fn run() -> Result<(), String> {
        use ahash::AHashMap;
        use kbnf::{Engine, EngineLike, Grammar, Token, Vocabulary};
        let grammar_str = r##"
start ::= #e"(.|\n)*\n\n";
"##;
        let mut token_strings: AHashMap<u32, String> = AHashMap::default();
        token_strings.extend([
            (1, "a".to_string()),
            (2, "hello".to_string()),
            (4, "\n".to_string()),
            (5, "\n\n".to_string()),
        ]);
        let tokens = token_strings
            .iter()
            .map(|(k, v)| (*k, Token(v.as_bytes().to_vec().into_boxed_slice())))
            .collect::<AHashMap<u32, _>>();
        let vocab = Vocabulary::new(tokens, token_strings).unwrap();
        let mut engine = Engine::new(grammar_str, vocab).unwrap();
        let mut logits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // The logits of the language model
        engine.compute_allowed_token_ids();
        assert_eq!(
            engine
                .allowed_token_ids_from_last_computation()
                .ones()
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 5]
        );
        engine.mask_logits(&mut logits).unwrap(); // mask the logits
        assert_eq!(&format!("{:?}", logits), "[-inf, 0.0, 0.0, -inf, 0.0, 0.0]");

        Ok(())
    }
}

symphony::main_sync!(ConstrainedDecoding);
