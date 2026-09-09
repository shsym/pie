use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::glm::STOP_TOKENS],
    pinned: &[],
};

pub const CONTRACT_VISION: Contract = Contract {
    markers: &[
        chat_template::glm::STOP_TOKENS,
        super::media::VISION_DELIMITERS,
    ],
    pinned: &[],
};
