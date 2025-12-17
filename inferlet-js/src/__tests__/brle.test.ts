import { describe, it, expect } from 'vitest';
import { Brle, causalMask } from '../brle.js';

describe('Brle', () => {
  describe('new()', () => {
    it('should create an empty Brle for size 0', () => {
      const brle = Brle.new(0);
      expect(brle.len()).toBe(0);
      expect(brle.isEmpty()).toBe(true);
      expect(brle.buffer).toEqual([]);
    });

    it('should create a Brle with all false values (visible positions)', () => {
      const brle = Brle.new(5);
      expect(brle.len()).toBe(5);
      expect(brle.isEmpty()).toBe(false);
      // [5] means 5 false values (visible positions, matching Rust semantics)
      expect(brle.buffer).toEqual([5]);
      expect(brle.toArray()).toEqual([false, false, false, false, false]);
    });
  });

  describe('fromArray()', () => {
    it('should encode [false, false, true, true, true, false]', () => {
      const brle = Brle.fromArray([false, false, true, true, true, false]);
      expect(brle.len()).toBe(6);
      // [2, 3, 1] means 2 false, 3 true, 1 false
      expect(brle.buffer).toEqual([2, 3, 1]);
    });

    it('should encode [true, true, false]', () => {
      const brle = Brle.fromArray([true, true, false]);
      expect(brle.len()).toBe(3);
      // [0, 2, 1] means 0 false, 2 true, 1 false
      expect(brle.buffer).toEqual([0, 2, 1]);
    });

    it('should encode all false', () => {
      const brle = Brle.fromArray([false, false, false]);
      expect(brle.len()).toBe(3);
      expect(brle.buffer).toEqual([3]);
    });

    it('should encode all true', () => {
      const brle = Brle.fromArray([true, true, true]);
      expect(brle.len()).toBe(3);
      expect(brle.buffer).toEqual([0, 3]);
    });

    it('should handle empty array', () => {
      const brle = Brle.fromArray([]);
      expect(brle.len()).toBe(0);
      expect(brle.isEmpty()).toBe(true);
    });
  });

  describe('toArray()', () => {
    it('should decode correctly', () => {
      const original = [false, true, true, false, false, true];
      const brle = Brle.fromArray(original);
      expect(brle.toArray()).toEqual(original);
    });

    it('should round-trip complex patterns', () => {
      const patterns = [
        [true],
        [false],
        [true, false, true, false],
        [false, false, false, true, true, true],
        Array(100).fill(true),
        Array(100).fill(false),
      ];

      for (const pattern of patterns) {
        const brle = Brle.fromArray(pattern);
        expect(brle.toArray()).toEqual(pattern);
      }
    });
  });

  describe('append()', () => {
    it('should append to empty Brle', () => {
      const brle = Brle.new(0);
      brle.append(true);
      expect(brle.len()).toBe(1);
      expect(brle.toArray()).toEqual([true]);
    });

    it('should append false to empty Brle', () => {
      const brle = Brle.new(0);
      brle.append(false);
      expect(brle.len()).toBe(1);
      expect(brle.toArray()).toEqual([false]);
    });

    it('should extend existing run', () => {
      const brle = Brle.fromArray([true, true]);
      brle.append(true);
      expect(brle.len()).toBe(3);
      expect(brle.toArray()).toEqual([true, true, true]);
    });

    it('should start new run when value changes', () => {
      const brle = Brle.fromArray([true, true]);
      brle.append(false);
      expect(brle.len()).toBe(3);
      expect(brle.toArray()).toEqual([true, true, false]);
    });
  });

  describe('maskRange()', () => {
    it('should mask a range to true', () => {
      const brle = Brle.fromArray([false, false, false, false, false]);
      brle.maskRange(1, 4, true);
      expect(brle.toArray()).toEqual([false, true, true, true, false]);
    });

    it('should mask a range to false', () => {
      const brle = Brle.fromArray([true, true, true, true, true]);
      brle.maskRange(1, 4, false);
      expect(brle.toArray()).toEqual([true, false, false, false, true]);
    });

    it('should handle empty range', () => {
      const brle = Brle.fromArray([true, true, true]);
      brle.maskRange(2, 2, false);
      expect(brle.toArray()).toEqual([true, true, true]);
    });

    it('should handle range at start', () => {
      const brle = Brle.fromArray([false, false, false]);
      brle.maskRange(0, 2, true);
      expect(brle.toArray()).toEqual([true, true, false]);
    });

    it('should handle range at end', () => {
      const brle = Brle.fromArray([false, false, false]);
      brle.maskRange(1, 3, true);
      expect(brle.toArray()).toEqual([false, true, true]);
    });
  });

  describe('mask()', () => {
    it('should mask individual indices', () => {
      const brle = Brle.fromArray([false, false, false, false, false]);
      brle.mask([0, 2, 4], true);
      expect(brle.toArray()).toEqual([true, false, true, false, true]);
    });

    it('should handle unsorted indices', () => {
      const brle = Brle.fromArray([false, false, false, false, false]);
      brle.mask([4, 0, 2], true);
      expect(brle.toArray()).toEqual([true, false, true, false, true]);
    });

    it('should handle duplicate indices', () => {
      const brle = Brle.fromArray([false, false, false]);
      brle.mask([1, 1, 1], true);
      expect(brle.toArray()).toEqual([false, true, false]);
    });

    it('should handle contiguous indices (merges into range)', () => {
      const brle = Brle.fromArray([false, false, false, false, false]);
      brle.mask([1, 2, 3], true);
      expect(brle.toArray()).toEqual([false, true, true, true, false]);
    });
  });

  describe('clone()', () => {
    it('should create an independent copy', () => {
      const original = Brle.fromArray([true, false, true]);
      const cloned = original.clone();

      // Modify original
      original.maskRange(0, 1, false);

      // Clone should be unchanged
      expect(cloned.toArray()).toEqual([true, false, true]);
      expect(original.toArray()).toEqual([false, false, true]);
    });
  });

  describe('iterRuns()', () => {
    it('should iterate over runs', () => {
      const brle = Brle.fromArray([false, false, true, true, true, false]);
      const runs = [...brle.iterRuns()];
      expect(runs).toEqual([
        [false, 0, 2],  // 2 false values from index 0 to 2
        [true, 2, 5],   // 3 true values from index 2 to 5
        [false, 5, 6],  // 1 false value from index 5 to 6
      ]);
    });

    it('should handle all-false sequence', () => {
      const brle = Brle.new(5);
      const runs = [...brle.iterRuns()];
      expect(runs).toEqual([
        [false, 0, 5],  // 5 false values (visible positions)
      ]);
    });
  });
});

describe('causalMask', () => {
  it('should create correct masks for single token', () => {
    const masks = causalMask(1, 1);
    expect(masks.length).toBe(1);
    // First token can only see itself (false = visible/can attend)
    expect(masks[0].len()).toBe(1);
    expect(masks[0].toArray()).toEqual([false]);
  });

  it('should create correct masks for multiple tokens', () => {
    const masks = causalMask(3, 3);
    expect(masks.length).toBe(3);

    // Token 0 can see position 0 (false = visible)
    expect(masks[0].toArray()).toEqual([false]);
    // Token 1 can see positions 0, 1
    expect(masks[1].toArray()).toEqual([false, false]);
    // Token 2 can see positions 0, 1, 2
    expect(masks[2].toArray()).toEqual([false, false, false]);
  });

  it('should handle offset correctly', () => {
    // Total 5 tokens, but only processing last 2
    const masks = causalMask(5, 2);
    expect(masks.length).toBe(2);

    // First new token (at position 3) can see positions 0-3 (false = visible)
    expect(masks[0].len()).toBe(4);
    expect(masks[0].toArray()).toEqual([false, false, false, false]);

    // Second new token (at position 4) can see positions 0-4
    expect(masks[1].len()).toBe(5);
    expect(masks[1].toArray()).toEqual([false, false, false, false, false]);
  });
});
