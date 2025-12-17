import { describe, it, expect } from 'vitest';

// Note: We test pure TypeScript exports here. Modules that depend on WIT bindings
// (runtime, model, tokenizer) can only be tested when running in WASM context.

// Import pure TypeScript modules directly (not via index.ts which imports binding-dependent modules)
import { Sampler } from '../sampler.js';
import { maxLen, endsWithAny } from '../stop-condition.js';
import { ChatFormatter } from '../chat.js';

describe('inferlet exports (pure TypeScript)', () => {
  describe('Sampler', () => {
    it('should be importable and create instances', () => {
      const sampler = Sampler.greedy();
      expect(sampler).toBeDefined();
      expect(sampler.getConfig().type).toBe('Multinomial');
    });

    it('should support various sampler types', () => {
      const topP = Sampler.topP(0.8, 0.95);
      expect(topP.getConfig().type).toBe('TopP');

      const topK = Sampler.topK(0.8, 40);
      expect(topK.getConfig().type).toBe('TopK');
    });
  });

  describe('StopCondition', () => {
    it('should be importable and create conditions', () => {
      const condition = maxLen(100);
      expect(condition).toBeDefined();
      expect(condition.check([])).toBe(false);
    });

    it('should support endsWithAny', () => {
      const condition = endsWithAny([[1, 2], [3, 4]]);
      expect(condition.check([5, 6, 1, 2])).toBe(true);
      expect(condition.check([5, 6, 7])).toBe(false);
    });
  });

  describe('ChatFormatter', () => {
    it('should be importable and format messages', () => {
      const formatter = new ChatFormatter();
      formatter.user('Hello');
      // Use proper template syntax with required parameters
      const template = '{% for msg in messages %}{{ msg.content }}{% endfor %}';
      const result = formatter.render(template, false, false);
      expect(result).toBe('Hello');
    });
  });
});
