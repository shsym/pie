use inferlet2::sampler::Sampler;
use inferlet2::traits::Tokenize;
use inferlet2::traits::tokenize::Tokenizer;
use llguidance::api::GrammarInit;
use llguidance::earley::SlicedBiasComputer;
use llguidance::{
    Constraint, ParserFactory,
    api::TopLevelGrammar,
    toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
};
use std::sync::Arc;
use std::time::Instant;

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

struct TrieTokenizer {
    tok_trie: TokTrie,
}

impl TrieTokenizer {
    fn new(
        words: &Vec<Vec<u8>>,
        tok_eos: u32,
        tok_bos: Option<u32>,
        tok_pad: Option<u32>,
        tok_unk: Option<u32>,
        tok_end_of_turn: Option<u32>,
    ) -> Self {
        let info = TokRxInfo {
            vocab_size: words.len() as u32,
            tok_eos,
            tok_bos,
            tok_pad,
            tok_unk,
            tok_end_of_turn,
        };
        let tok_trie = TokTrie::from(&info, &words);
        TrieTokenizer { tok_trie }
    }

    fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for TrieTokenizer {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.greedy_tokenize(s)
    }
}

fn get_parser_factory(env: &TokEnv) -> ParserFactory {
    let mut fact = ParserFactory::new(
        &env,
        InferenceCapabilities {
            ff_tokens: false,             // can the engine append multiple tokens?
            backtrack: false,             // can the engine remove generated tokens?
            conditional_ff_tokens: false, // not used
            fork: false,                  // not used
        },
        &SlicedBiasComputer::general_slices(),
    )
    .unwrap();
    fact.set_stderr_log_level(2);
    fact.set_buffer_log_level(0);
    fact
}

struct ConstrainedSampler {
    constraint: Constraint,
}

impl ConstrainedSampler {
    pub fn new(
        tokenizer: Tokenizer,
        lark: &str,
        eos_token: &str,
        bos_token: Option<&str>,
        pad_token: Option<&str>,
        unk_token: Option<&str>,
        end_of_turn_token: Option<&str>,
    ) -> Self {
        //let words = l4m::get_vocabs();
        let words = tokenizer.get_vocabs();
        let eos_token = tokenizer.tokenize(eos_token)[0];
        let bos_token = bos_token.map(|s| tokenizer.tokenize(s)[0]);
        let pad_token = pad_token.map(|s| tokenizer.tokenize(s)[0]);
        let unk_token = unk_token.map(|s| tokenizer.tokenize(s)[0]);
        let end_of_turn_token = end_of_turn_token.map(|s| tokenizer.tokenize(s)[0]);

        let tokenizer = TrieTokenizer::new(
            &words,
            eos_token,
            bos_token,
            pad_token,
            unk_token,
            end_of_turn_token,
        )
        .to_env();

        let grm = TopLevelGrammar::from_lark(lark.to_string());

        let quiet = true;

        let parser = get_parser_factory(&tokenizer)
            .create_parser_from_init(
                GrammarInit::Serialized(grm),
                if quiet { 0 } else { 2 },
                if quiet { 1 } else { 2 },
            )
            .unwrap();

        let constraint = Constraint::new(parser);
        ConstrainedSampler { constraint }
    }
}

impl Sampler for ConstrainedSampler {
    fn sample(&mut self, token_ids: &[u32], logits: &[f32]) -> u32 {
        let res = self.constraint.compute_mask().unwrap();

        let sampled_token_id = if let Some(mask) = &res.sample_mask {
            let mut max_logit = f32::NEG_INFINITY;
            let mut max_logit_idx = 0;
            let mut sampled = false;

            // traverse the token_ids and return the first token_id that is allowed by the mask
            for (i, token_id) in token_ids.iter().enumerate() {
                let token_id = *token_id as TokenId;
                let logit = logits[i];
                if mask.is_allowed(token_id) {
                    sampled = true;
                    if logit > max_logit {
                        max_logit = logit;
                        max_logit_idx = i;
                    }
                }
            }

            // check if any token was selected
            if !sampled {
                //println!("No token selected");
                // if no token was selected, then select one allowed by the mask
                mask.first_bit_set().unwrap() as u32
            } else {
                //println!("Token selected: {:?}", max_logit_idx);
                token_ids[max_logit_idx]
            }
        } else {
            //println!("No mask found");
            token_ids[0]
        };

        self.constraint
            .commit_token(Some(sampled_token_id))
            .unwrap();
        sampled_token_id
    }
}

#[inferlet2::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();

    let model = inferlet2::get_auto_model();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    let prompt = "what is the capital of France? output data";

    let mut sampler = ConstrainedSampler::new(
        tokenizer.clone(),
        JSON_GRAMMAR,
        "<|end_of_text|>",
        Some("<|begin_of_text|>"),
        None,
        None,
        Some("<|eot_id|>"),
    );
    let mut stop_condition = inferlet2::stop_condition::any(
        inferlet2::stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        inferlet2::stop_condition::Length::new(128),
    );
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\n");
    ctx.fill(&prompt);
    ctx.fill("<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let output = ctx.generate(&mut sampler, &mut stop_condition).await;
    let output_token_ids = tokenizer.tokenize(&output);

    println!(
        "Output: {:?} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    // compute per token latency
    println!(
        "Per token latency: {:?}",
        start.elapsed() / output_token_ids.len() as u32
    );

    Ok(())
}
