use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::deepseek::STOP_TOKENS],
    pinned: &[],
};
