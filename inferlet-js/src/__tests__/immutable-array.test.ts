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

  it('should support pushAll with multiple items', () => {
    const arr = ImmutableArray.from([1, 2]);
    const result = arr.pushAll([3, 4, 5]);

    expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    expect(result.length).toBe(5);
    // Original should be unchanged
    expect(arr.toArray()).toEqual([1, 2]);
  });

  it('should return same instance for pushAll with empty array', () => {
    const arr = ImmutableArray.from([1, 2, 3]);
    const result = arr.pushAll([]);

    expect(result).toBe(arr);
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it('should create empty array', () => {
    const arr = ImmutableArray.empty<number>();

    expect(arr.length).toBe(0);
    expect(arr.toArray()).toEqual([]);
  });

  it('should support push on empty array', () => {
    const arr = ImmutableArray.empty<number>();
    const result = arr.push(42);

    expect(result.toArray()).toEqual([42]);
    expect(arr.length).toBe(0); // Original unchanged
  });

  it('should support pushAll on empty array', () => {
    const arr = ImmutableArray.empty<number>();
    const result = arr.pushAll([1, 2, 3]);

    expect(result.toArray()).toEqual([1, 2, 3]);
    expect(arr.length).toBe(0); // Original unchanged
  });

  it('should support iteration', () => {
    const arr = ImmutableArray.from([1, 2, 3]);
    const collected: number[] = [];

    for (const item of arr) {
      collected.push(item);
    }

    expect(collected).toEqual([1, 2, 3]);
  });

  it('should support map', () => {
    const arr = ImmutableArray.from([1, 2, 3]);
    const doubled = arr.map((x) => x * 2);

    expect(doubled).toEqual([2, 4, 6]);
  });

  it('should handle out of bounds get', () => {
    const arr = ImmutableArray.from([1, 2, 3]);

    expect(arr.get(100)).toBeUndefined();
    expect(arr.get(-100)).toBeUndefined();
  });
});
