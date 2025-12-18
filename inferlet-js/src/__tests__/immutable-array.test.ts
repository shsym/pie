import { describe, it, expect } from 'vitest';
import { ImmutableArray } from '../immutable-array.js';

describe('ImmutableArray', () => {
  it('should create from array', () => {
    const arr = ImmutableArray.from([1, 2, 3]);
    expect(arr.toArray()).toEqual([1, 2, 3]);
    expect(arr.length).toBe(3);
  });

  it('should share underlying array on fork', () => {
    const original = ImmutableArray.from([1, 2, 3]);
    const forked = original.fork();

    // Both should have same values
    expect(forked.toArray()).toEqual([1, 2, 3]);

    // Mutating forked should not affect original
    const mutated = forked.push(4);
    expect(mutated.toArray()).toEqual([1, 2, 3, 4]);
    expect(original.toArray()).toEqual([1, 2, 3]);
  });

  it('should support slice', () => {
    const arr = ImmutableArray.from([1, 2, 3, 4, 5]);
    const sliced = arr.slice(1, 3);
    expect(sliced.toArray()).toEqual([2, 3]);
  });

  it('should support concat', () => {
    const arr1 = ImmutableArray.from([1, 2]);
    const arr2 = [3, 4];
    const combined = arr1.concat(arr2);
    expect(combined.toArray()).toEqual([1, 2, 3, 4]);
  });

  it('should support get by index', () => {
    const arr = ImmutableArray.from([10, 20, 30]);
    expect(arr.get(0)).toBe(10);
    expect(arr.get(1)).toBe(20);
    expect(arr.get(2)).toBe(30);
    expect(arr.get(-1)).toBe(30);
  });
});
