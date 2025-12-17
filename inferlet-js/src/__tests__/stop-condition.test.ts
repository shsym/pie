import { describe, it, expect } from 'vitest';
import {
  StopCondition,
  MaxLen,
  EndsWith,
  AnyEndsWith,
  Or,
  maxLen,
  endsWith,
  endsWithAny,
} from '../stop-condition.js';

describe('StopCondition', () => {
  describe('MaxLen', () => {
    it('should stop when sequence reaches max length', () => {
      const condition = new MaxLen(5);

      expect(condition.check([1, 2, 3, 4])).toBe(false);
      expect(condition.check([1, 2, 3, 4, 5])).toBe(true);
      expect(condition.check([1, 2, 3, 4, 5, 6])).toBe(true);
    });

    it('should handle edge cases', () => {
      const condition = new MaxLen(0);
      expect(condition.check([])).toBe(true);
      expect(condition.check([1])).toBe(true);
    });

    it('should be created via factory function', () => {
      const condition = maxLen(10);
      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).toBe(true);
    });
  });

  describe('EndsWith', () => {
    it('should stop when sequence ends with specific tokens', () => {
      const condition = new EndsWith([100, 101]);

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 100, 101])).toBe(true);
      expect(condition.check([100, 101])).toBe(true);
      expect(condition.check([100, 101, 102])).toBe(false);
    });

    it('should handle single token sequences', () => {
      const condition = new EndsWith([42]);

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 42])).toBe(true);
      expect(condition.check([42])).toBe(true);
    });

    it('should handle empty token sequence', () => {
      const condition = new EndsWith([]);
      expect(condition.check([])).toBe(true);
      expect(condition.check([1, 2, 3])).toBe(true);
    });

    it('should not match if sequence is too short', () => {
      const condition = new EndsWith([1, 2, 3]);
      expect(condition.check([1, 2])).toBe(false);
      expect(condition.check([1])).toBe(false);
      expect(condition.check([])).toBe(false);
    });

    it('should be created via factory function', () => {
      const condition = endsWith([50256]);
      expect(condition.check([1, 2, 50256])).toBe(true);
      expect(condition.check([1, 2, 3])).toBe(false);
    });
  });

  describe('AnyEndsWith', () => {
    it('should stop when sequence ends with any of the specified sequences', () => {
      const condition = new AnyEndsWith([
        [100, 101],
        [200, 201],
        [42],
      ]);

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 100, 101])).toBe(true);
      expect(condition.check([1, 2, 200, 201])).toBe(true);
      expect(condition.check([1, 2, 42])).toBe(true);
      expect(condition.check([1, 2, 100])).toBe(false);
    });

    it('should handle empty sequences list', () => {
      const condition = new AnyEndsWith([]);
      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([])).toBe(false);
    });

    it('should handle single sequence', () => {
      const condition = new AnyEndsWith([[100, 101]]);
      expect(condition.check([1, 2, 100, 101])).toBe(true);
      expect(condition.check([1, 2, 3])).toBe(false);
    });

    it('should be created via factory function', () => {
      const condition = endsWithAny([[50256], [50257], [100, 101]]);
      expect(condition.check([1, 2, 50256])).toBe(true);
      expect(condition.check([1, 2, 50257])).toBe(true);
      expect(condition.check([1, 2, 100, 101])).toBe(true);
      expect(condition.check([1, 2, 3])).toBe(false);
    });
  });

  describe('Or combinator', () => {
    it('should stop when either condition is met', () => {
      const condition = new Or(new MaxLen(5), new EndsWith([42]));

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 3, 4, 5])).toBe(true); // MaxLen
      expect(condition.check([1, 2, 42])).toBe(true); // EndsWith
      expect(condition.check([1, 2, 3, 4, 5, 42])).toBe(true); // Both
    });

    it('should support chaining via or() method', () => {
      const condition = maxLen(100).or(endsWith([50256]));

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 50256])).toBe(true);

      const tokens = new Array(100).fill(1);
      expect(condition.check(tokens)).toBe(true);
    });

    it('should support multiple or() chaining', () => {
      const condition = maxLen(100)
        .or(endsWith([50256]))
        .or(endsWith([50257]));

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 50256])).toBe(true);
      expect(condition.check([1, 2, 50257])).toBe(true);

      const tokens = new Array(100).fill(1);
      expect(condition.check(tokens)).toBe(true);
    });

    it('should work with AnyEndsWith', () => {
      const condition = maxLen(50).or(endsWithAny([[100], [200], [300]]));

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 100])).toBe(true);
      expect(condition.check([1, 2, 200])).toBe(true);
      expect(condition.check([1, 2, 300])).toBe(true);

      const tokens = new Array(50).fill(1);
      expect(condition.check(tokens)).toBe(true);
    });
  });

  describe('Complex scenarios', () => {
    it('should handle realistic stop conditions for text generation', () => {
      // Stop at max 512 tokens OR when encountering EOS token (50256)
      const condition = maxLen(512).or(endsWith([50256]));

      const shortSequence = [1, 2, 3, 4, 5];
      expect(condition.check(shortSequence)).toBe(false);

      const sequenceWithEOS = [1, 2, 3, 50256];
      expect(condition.check(sequenceWithEOS)).toBe(true);

      const longSequence = new Array(512).fill(1);
      expect(condition.check(longSequence)).toBe(true);
    });

    it('should handle multiple stop sequences', () => {
      // Stop when encountering any common delimiter tokens
      const condition = endsWithAny([
        [198], // newline
        [13], // carriage return
        [50256], // EOS
      ]).or(maxLen(1000));

      expect(condition.check([1, 2, 3])).toBe(false);
      expect(condition.check([1, 2, 198])).toBe(true);
      expect(condition.check([1, 2, 13])).toBe(true);
      expect(condition.check([1, 2, 50256])).toBe(true);

      const longSequence = new Array(1000).fill(1);
      expect(condition.check(longSequence)).toBe(true);
    });

    it('should handle multi-token stop sequences', () => {
      // Stop when encountering specific multi-token patterns
      const condition = endsWithAny([
        [1, 2, 3], // pattern 1
        [4, 5, 6, 7], // pattern 2
      ]);

      expect(condition.check([1, 2])).toBe(false);
      expect(condition.check([4, 5, 6])).toBe(false);
      expect(condition.check([10, 11, 1, 2, 3])).toBe(true);
      expect(condition.check([10, 11, 4, 5, 6, 7])).toBe(true);
      expect(condition.check([1, 2, 3, 4])).toBe(false);
    });
  });

  describe('Type compatibility', () => {
    it('should allow StopCondition interface usage', () => {
      const conditions: StopCondition[] = [
        maxLen(100),
        endsWith([42]),
        endsWithAny([[1], [2], [3]]),
        maxLen(50).or(endsWith([100])),
      ];

      const tokenIds = [1, 2, 3];
      conditions.forEach((condition) => {
        expect(typeof condition.check(tokenIds)).toBe('boolean');
      });
    });

    it('should support or() method on all implementations', () => {
      const maxLenCondition = maxLen(10);
      const endsWithCondition = endsWith([42]);
      const anyEndsWithCondition = endsWithAny([[1], [2]]);

      expect(maxLenCondition.or(endsWithCondition)).toBeInstanceOf(Or);
      expect(endsWithCondition.or(maxLenCondition)).toBeInstanceOf(Or);
      expect(anyEndsWithCondition.or(maxLenCondition)).toBeInstanceOf(Or);
    });
  });
});
