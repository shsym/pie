// inferlet-js/test/mocks/inferlet-core-tokenize.js
// Mock implementation of inferlet:core/tokenize

// Simple character-based tokenizer for testing
const VOCAB_SIZE = 32000;
const SPECIAL_TOKENS = {
  '<|begin_of_text|>': 128000,
  '<|end_of_text|>': 128001,
  '<|eot_id|>': 128009,
  '<|start_header_id|>': 128006,
  '<|end_header_id|>': 128007,
};

export class Tokenizer {
  #model;

  constructor(model) {
    this.#model = model;
  }

  tokenize(text) {
    // Simple mock: each character becomes a token
    // Special tokens are handled specially
    const tokens = [];
    let i = 0;
    while (i < text.length) {
      let matched = false;
      // Check for special tokens
      for (const [special, id] of Object.entries(SPECIAL_TOKENS)) {
        if (text.slice(i).startsWith(special)) {
          tokens.push(id);
          i += special.length;
          matched = true;
          break;
        }
      }
      if (!matched) {
        // Simple character tokenization
        tokens.push(text.charCodeAt(i) % VOCAB_SIZE);
        i++;
      }
    }
    return new Uint32Array(tokens);
  }

  detokenize(tokens) {
    const chars = [];
    for (const token of tokens) {
      // Check if it's a special token
      let found = false;
      for (const [special, id] of Object.entries(SPECIAL_TOKENS)) {
        if (token === id) {
          chars.push(special);
          found = true;
          break;
        }
      }
      if (!found && token < 128) {
        chars.push(String.fromCharCode(token));
      } else if (!found) {
        chars.push(`[${token}]`);
      }
    }
    return chars.join('');
  }

  getVocabs() {
    // Return simplified vocab: [token_ids, token_bytes]
    const ids = new Uint32Array(256);
    const bytes = [];
    for (let i = 0; i < 256; i++) {
      ids[i] = i;
      bytes.push(new Uint8Array([i]));
    }
    return [ids, bytes];
  }

  getSplitRegex() {
    return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  }

  getSpecialTokens() {
    const ids = new Uint32Array(Object.values(SPECIAL_TOKENS));
    const bytes = Object.keys(SPECIAL_TOKENS).map(s =>
      new Uint8Array(s.split('').map(c => c.charCodeAt(0)))
    );
    return [ids, bytes];
  }
}

export function getTokenizer(model) {
  return new Tokenizer(model);
}
