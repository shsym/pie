// Tokenizer class for text tokenization and detokenization
// Mirrors the Rust Tokenizer from inferlet/src/lib.rs

import { Tokenizer as TokenizerBinding, getTokenizer } from './bindings/interfaces/inferlet-core-tokenize.js';
import type { Model } from './bindings/interfaces/inferlet-core-common.js';

/**
 * Tokenizer class that wraps the WIT tokenize bindings
 * Provides methods to convert text to token IDs and vice versa
 */
export class Tokenizer {
  private readonly inner: TokenizerBinding;

  /**
   * Creates a new Tokenizer instance for the given model
   * @param model - The Model instance to create a tokenizer for
   */
  constructor(model: Model) {
    this.inner = getTokenizer(model);
  }

  /**
   * Converts a string of text into a sequence of token IDs.
   *
   * @param text - The input string to tokenize
   * @returns A Uint32Array containing the corresponding token IDs
   */
  tokenize(text: string): Uint32Array {
    return this.inner.tokenize(text);
  }

  /**
   * Converts a sequence of token IDs back into a human-readable string.
   *
   * @param tokens - A Uint32Array of token IDs to detokenize
   * @returns The reconstructed string
   */
  detokenize(tokens: Uint32Array): string {
    return this.inner.detokenize(tokens);
  }

  /**
   * Retrieves the entire vocabulary of the tokenizer.
   *
   * @returns A tuple containing:
   *   - A Uint32Array of token IDs
   *   - An array of Uint8Array byte sequences representing the tokens
   */
  getVocabs(): [Uint32Array, Array<Uint8Array>] {
    return this.inner.getVocabs();
  }

  /**
   * Retrieves the special tokens recognized by the tokenizer.
   *
   * @returns A tuple containing:
   *   - A Uint32Array of special token IDs
   *   - An array of Uint8Array byte sequences representing the special tokens
   */
  getSpecialTokens(): [Uint32Array, Array<Uint8Array>] {
    return this.inner.getSpecialTokens();
  }

  /**
   * Retrieves the split regular expression used by the tokenizer.
   *
   * @returns A string representing the split regular expression
   */
  getSplitRegex(): string {
    return this.inner.getSplitRegex();
  }
}
