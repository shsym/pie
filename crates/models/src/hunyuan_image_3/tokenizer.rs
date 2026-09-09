use ::tokenizer::contract::Contract;

pub const END_OF_TEXT: &str = "<|endoftext|>";
pub const START_OF_TEXT: &str = "<|startoftext|>";

pub const STOP_TOKENS: &[&str] = &[END_OF_TEXT, "</answer>"];

pub const IMAGE_TOKENS: &[&str] = &[
    "<boi>",
    "<eoi>",
    "<img>",
    "<cfg>",
    "<timestep>",
    "<guidance>",
    "<joint_img_sep>",
    "<img_size_1024>",
    "<img_ratio_0>",
];

pub const STAGE_TOKENS: &[&str] = &[
    "<think>",
    "</think>",
    "<recaption>",
    "</recaption>",
    "<answer>",
];

pub const CONTRACT: Contract = Contract {
    markers: &[STOP_TOKENS, IMAGE_TOKENS, STAGE_TOKENS, &[START_OF_TEXT]],
    pinned: &[("<boi>", 128_000), ("<eoi>", 128_001), ("<img>", 128_006)],
};
