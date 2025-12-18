// inferlet-js/test/wasm/__tests__/tokenizer.wasm.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { __resetMockState as resetRuntime, __setMockModels } from '../../mocks/inferlet-core-runtime.js';
import { getAutoModel } from '../../../src/model.js';

describe('Tokenizer (WASM mock integration)', () => {
  beforeEach(() => {
    resetRuntime();
    __setMockModels(['mock-llama-3.2-1b']);
  });

  describe('tokenize()', () => {
    it('should tokenize text to token IDs', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const tokens = tokenizer.tokenize('Hello');
      expect(tokens).toBeInstanceOf(Uint32Array);
      expect(tokens.length).toBe(5); // 'H', 'e', 'l', 'l', 'o' - mock uses character tokenization
    });

    it('should handle special tokens', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const tokens = tokenizer.tokenize('<|eot_id|>');
      expect(tokens.length).toBe(1);
      expect(tokens[0]).toBe(128009); // Special token ID from mock
    });

    it('should handle mixed text and special tokens', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const tokens = tokenizer.tokenize('Hi<|eot_id|>');
      // 'H', 'i' = 2 chars + 1 special token = 3 total
      expect(tokens.length).toBe(3);
      expect(tokens[2]).toBe(128009);
    });
  });

  describe('detokenize()', () => {
    it('should convert tokens back to text', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const tokens = new Uint32Array([72, 101, 108, 108, 111]); // ASCII for "Hello"
      const text = tokenizer.detokenize(tokens);
      expect(text).toBe('Hello');
    });

    it('should handle special tokens in detokenization', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const tokens = new Uint32Array([128009]); // <|eot_id|>
      const text = tokenizer.detokenize(tokens);
      expect(text).toBe('<|eot_id|>');
    });
  });

  describe('round-trip', () => {
    it('should round-trip simple ASCII text', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const original = 'Test';
      const tokens = tokenizer.tokenize(original);
      const decoded = tokenizer.detokenize(tokens);
      expect(decoded).toBe(original);
    });

    it('should round-trip special tokens', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const original = '<|eot_id|>';
      const tokens = tokenizer.tokenize(original);
      const decoded = tokenizer.detokenize(tokens);
      expect(decoded).toBe(original);
    });
  });

  describe('vocabulary and special tokens', () => {
    it('should return vocabulary', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const [ids, bytes] = tokenizer.vocabs;
      expect(ids).toBeInstanceOf(Uint32Array);
      expect(Array.isArray(bytes)).toBe(true);
      expect(ids.length).toBe(bytes.length);
    });

    it('should return special tokens', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const [ids, bytes] = tokenizer.specialTokens;
      expect(ids).toBeInstanceOf(Uint32Array);
      expect(Array.isArray(bytes)).toBe(true);
      expect(ids.length).toBeGreaterThan(0);
    });

    it('should return split regex', () => {
      const model = getAutoModel();
      const tokenizer = model.tokenizer;

      const regex = tokenizer.splitRegex;
      expect(typeof regex).toBe('string');
      expect(regex.length).toBeGreaterThan(0);
    });
  });
});
