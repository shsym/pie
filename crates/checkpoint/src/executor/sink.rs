use std::collections::HashMap;

use crate::error::Error;

pub trait TensorSink {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error>;
}

#[derive(Debug, Default)]
pub struct MemorySink {
    pub tensors: HashMap<String, Vec<u8>>,
}

impl TensorSink for MemorySink {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        self.tensors.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }
}
