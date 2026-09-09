use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::kimi::STOP_TOKENS],
    pinned: &[],
};
