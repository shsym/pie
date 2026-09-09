use ::tokenizer::contract::Contract;

pub const IM_END: &str = "<|im_end|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

pub const STOP_TOKENS: &[&str] = &[IM_END, END_OF_TEXT];

pub const VISION_START: &str = "<|vision_start|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VISION_END: &str = "<|vision_end|>";

pub const VISION_DELIMITERS: &[&str] = &[VISION_START, IMAGE_PAD, VISION_END];

pub const AUDIO_SPECIALS: &[(&str, u32)] = &[
    ("<|audio_start|>", 248_070),
    ("<|audio_end|>", 248_071),
    ("<tts_pad>", 248_072),
    ("<tts_text_bos>", 248_073),
    ("<tts_text_eod>", 248_074),
    ("<tts_text_bos_single>", 248_075),
    ("<|audio_pad|>", 248_076),
];

pub const CONTRACT: Contract = Contract {
    markers: &[STOP_TOKENS],
    pinned: &[],
};

pub const CONTRACT_VISION: Contract = Contract {
    markers: &[STOP_TOKENS, VISION_DELIMITERS],
    pinned: &[],
};

pub const CONTRACT_38: Contract = Contract {
    markers: &[STOP_TOKENS],
    pinned: AUDIO_SPECIALS,
};

pub const CONTRACT_38_VISION: Contract = Contract {
    markers: &[STOP_TOKENS, VISION_DELIMITERS],
    pinned: AUDIO_SPECIALS,
};
