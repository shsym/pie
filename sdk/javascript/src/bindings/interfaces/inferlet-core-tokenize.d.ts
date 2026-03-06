/** @module Interface inferlet:core/tokenize **/
/**
 * Retrieves the tokenizer associated with the given model
 */
export function getTokenizer(model: Model): Tokenizer;
export type Model = import('./inferlet-core-common.js').Model;

export class Tokenizer {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Converts input text into a list of token IDs
  */
  tokenize(text: string): Uint32Array;
  /**
  * Converts token IDs back into a decoded string
  */
  detokenize(tokens: Uint32Array): string;
  /**
  * Returns the tokenizer's vocabulary as a list of byte sequences (tokens)
  */
  getVocabs(): [Uint32Array, Array<Uint8Array>];
  /**
  * Returns the split regular expression used by the tokenizer
  */
  getSplitRegex(): string;
  /**
  * Returns the special tokens recognized by the tokenizer
  */
  getSpecialTokens(): [Uint32Array, Array<Uint8Array>];
}
