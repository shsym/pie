use ::tokenizer::contract::Contract;

pub const EOS: &str = "</s>";
pub const PAD: &str = "<pad>";
pub const UNK: &str = "<unk>";

pub const MARKERS: &[&str] = &[EOS, PAD, UNK];

pub const PINNED: &[(&str, u32)] = &[(PAD, 0), (EOS, 1), (UNK, 3)];

pub const CONTRACT: Contract = Contract {
    markers: &[MARKERS],
    pinned: PINNED,
};
