pub const META_PREFIX: &str = "__meta__/";

pub const VERSION_KEY: &str = "pie_version";

pub const SOURCE_KEY: &str = "pie_source";

pub const SOURCE_ENCODING_KEY: &str = "pie_source_encoding";

pub const RUNTIME_QUANT_KEY: &str = "pie_runtime_quant";

pub fn is_meta(name: &str) -> bool {
    name.starts_with(META_PREFIX)
}

pub fn meta_name(path: &str) -> String {
    format!("{META_PREFIX}{path}")
}

pub fn reject_reserved(name: &str) -> Result<(), crate::error::Error> {
    if is_meta(name) {
        return Err(crate::error::Error::Checkpoint(format!(
            "tensor {name:?} is in the reserved metadata namespace ({META_PREFIX}); \
             pie artifacts keep the compiled tokenizer and model descriptor there, \
             so a weight cannot be written under that prefix"
        )));
    }
    Ok(())
}
