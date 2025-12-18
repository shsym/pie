// Tokenizer class for text tokenization and detokenization
// Mirrors the Rust Tokenizer from inferlet/src/lib.rs

import { Tokenizer as TokenizerBinding, getTokenizer } from 'inferlet:core/tokenize';
import type { Model } from 'inferlet:core/common';

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
   * The entire vocabulary of the tokenizer.
   * Returns a tuple of [token IDs, byte sequences]
   */
  get vocabs(): [Uint32Array, Array<Uint8Array>] {
    return this.inner.getVocabs();
  }

  /**
   * The special tokens recognized by the tokenizer.
   * Returns a tuple of [special token IDs, byte sequences]
   */
  get specialTokens(): [Uint32Array, Array<Uint8Array>] {
    return this.inner.getSpecialTokens();
  }

  /**
   * The split regular expression used by the tokenizer.
   */
  get splitRegex(): string {
    return this.inner.getSplitRegex();
  }

  // Deprecated methods for backward compatibility
  /** @deprecated Use `vocabs` getter instead */
  getVocabs(): [Uint32Array, Array<Uint8Array>] { return this.vocabs; }
  /** @deprecated Use `specialTokens` getter instead */
  getSpecialTokens(): [Uint32Array, Array<Uint8Array>] { return this.specialTokens; }
  /** @deprecated Use `splitRegex` getter instead */
  getSplitRegex(): string { return this.splitRegex; }
}
