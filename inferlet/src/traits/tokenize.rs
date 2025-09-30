use crate::tokenize;
use crate::{Model, Queue};
use std::rc::Rc;

/// A handle to a text tokenizer engine.
///
/// This struct provides methods to convert between raw text and token IDs,
/// which are the standard input for language models. It's a lightweight,
/// cloneable wrapper that allows the underlying tokenizer resource to be
/// shared efficiently.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    inner: Rc<tokenize::Tokenizer>,
}

/// A trait for objects that can provide access to a `Tokenizer`.
///
/// This abstracts the source of the tokenizer, which is typically associated
/// with a specific model and managed by an execution context like a `Queue`.
pub trait Tokenize {
    /// Creates and returns a new `Tokenizer` handle.
    fn get_tokenizer(&self) -> Tokenizer;
}

impl Tokenizer {
    pub fn new(model: &Model) -> Tokenizer {
        Tokenizer {
            inner: Rc::new(tokenize::get_tokenizer(&model.inner)),
        }
    }

    /// Converts a string of text into a sequence of token IDs.
    ///
    /// # Parameters
    /// * `text`: The input string slice to tokenize.
    ///
    /// # Returns
    /// A `Vec<u32>` containing the corresponding token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tokenize(text)
    }

    /// Converts a sequence of token IDs back into a human-readable string.
    ///
    /// # Parameters
    /// * `tokens`: A slice of token IDs to detokenize.
    ///
    /// # Returns
    /// The reconstructed `String`.
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.inner.detokenize(tokens)
    }

    /// Retrieves the entire vocabulary of the tokenizer.
    ///
    /// # Returns
    /// A `Vec<Vec<u8>>`, where each inner vector represents a single token in the
    /// vocabulary as a sequence of bytes.
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.get_vocabs()
    }
}

impl Tokenize for Model {
    fn get_tokenizer(&self) -> Tokenizer {
        Tokenizer::new(self)
    }
}
