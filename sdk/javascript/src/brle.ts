// Binary Run-Length Encoding (BRLE) structure.
// This efficiently stores a large sequence of booleans by encoding
// the lengths of consecutive runs of `false` and `true` values.
// Mirrors the Rust Brle from inferlet/src/brle.rs

/**
 * A Binary Run-Length Encoding (BRLE) structure.
 *
 * This structure efficiently stores a large sequence of booleans by encoding
 * the lengths of consecutive runs of `false` and `true` values.
 * The sequence always begins with a run of `false`s, which may be zero-length.
 *
 * Encoding examples:
 * - `[false, false, true, true, true, false]` is encoded as `[2, 3, 1]`.
 * - `[true, true, false]` is encoded as `[0, 2, 1]`.
 */
export class Brle {
  /** The buffer of run lengths. Even indices are for `false` runs, odd for `true`. */
  buffer: number[];
  /** The total number of boolean values represented. */
  totalSize: number;

  private constructor(buffer: number[], totalSize: number) {
    this.buffer = buffer;
    this.totalSize = totalSize;
  }

  /**
   * Creates a new `Brle` instance representing `size` `false` values.
   * Matches Rust semantics where FALSE = visible (can attend).
   */
  static new(size: number): Brle {
    if (size === 0) {
      return new Brle([], 0);
    }
    // All false values: [size] means size false values (visible positions)
    return new Brle([size], size);
  }

  /**
   * Creates a `Brle` from an array of booleans.
   */
  static fromArray(values: boolean[]): Brle {
    if (values.length === 0) {
      return Brle.new(0);
    }

    const buffer: number[] = [];
    let currentVal = false;
    let count = 0;

    // Handle the initial run of `false`s, which could be zero-length.
    if (values[0]) {
      buffer.push(0);
      currentVal = true;
    }

    for (const val of values) {
      if (val === currentVal) {
        count++;
      } else {
        buffer.push(count);
        currentVal = val;
        count = 1;
      }
    }
    buffer.push(count); // Push the last run

    return new Brle(buffer, values.length);
  }

  /**
   * Returns the total number of booleans in the sequence.
   */
  len(): number {
    return this.totalSize;
  }

  /**
   * Returns `true` if the sequence is empty.
   */
  isEmpty(): boolean {
    return this.totalSize === 0;
  }

  /**
   * Decodes the `Brle` into an array of booleans.
   */
  toArray(): boolean[] {
    const result: boolean[] = [];
    for (const [value, start, end] of this.iterRuns()) {
      const runLen = end - start;
      for (let i = 0; i < runLen; i++) {
        result.push(value);
      }
    }
    return result;
  }

  /**
   * Appends a boolean value to the end of the sequence.
   */
  append(flag: boolean): void {
    if (this.buffer.length === 0) {
      if (flag) {
        this.buffer.push(0, 1);
      } else {
        this.buffer.push(1);
      }
    } else {
      const lastRunIsTrue = (this.buffer.length - 1) % 2 !== 0;
      if (lastRunIsTrue === flag) {
        this.buffer[this.buffer.length - 1]++;
      } else {
        this.buffer.push(1);
      }
    }
    this.totalSize++;
  }

  /**
   * Sets a range of booleans to a specified value.
   * The range is exclusive of `end` (`start..end`).
   */
  maskRange(start: number, end: number, flag: boolean): void {
    if (start >= end) {
      return;
    }
    this.maskInternal([[start, end]], flag);
  }

  /**
   * Sets multiple indices to a specified value.
   */
  mask(indices: number[], flag: boolean): void {
    if (indices.length === 0) {
      return;
    }

    // Sort and deduplicate indices
    const sortedIndices = [...new Set(indices)].sort((a, b) => a - b);

    // Group into contiguous ranges
    const ranges: [number, number][] = [];
    let rangeStart = sortedIndices[0];
    let rangeEnd = rangeStart + 1;

    for (let i = 1; i < sortedIndices.length; i++) {
      const index = sortedIndices[i];
      if (index === rangeEnd) {
        rangeEnd = index + 1;
      } else {
        ranges.push([rangeStart, rangeEnd]);
        rangeStart = index;
        rangeEnd = index + 1;
      }
    }
    ranges.push([rangeStart, rangeEnd]);

    this.maskInternal(ranges, flag);
  }

  /**
   * Returns an iterator over the runs, yielding [value, startIndex, endIndex].
   */
  *iterRuns(): Generator<[boolean, number, number]> {
    let currentPos = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const runLen = this.buffer[i];
      const value = i % 2 !== 0;
      const start = currentPos;
      const end = currentPos + runLen;
      currentPos = end;
      yield [value, start, end];
    }
  }

  /**
   * Checks the boolean values at a given set of indices.
   * Returns whether each index is masked (TRUE = masked/hidden from attention).
   *
   * This is optimized to check multiple indices in a single pass over the data.
   *
   * @param indices - An array of indices to check. The indices do not need to be sorted.
   * @returns An array of booleans containing the value at each corresponding index.
   */
  isMasked(indices: number[]): boolean[] {
    if (indices.length === 0) {
      return [];
    }

    // Pair indices with their original position for result ordering
    const indexedIndices: [number, number][] = indices.map((idx, i) => [i, idx]);
    indexedIndices.sort((a, b) => a[1] - b[1]);

    const results = new Array<boolean>(indices.length).fill(false);
    const runIterator = this.iterRuns();
    let currentRun = runIterator.next();

    for (const [originalPos, queryIndex] of indexedIndices) {
      if (queryIndex >= this.totalSize) {
        throw new Error(
          `Index ${queryIndex} is out of bounds for Brle of length ${this.totalSize}`
        );
      }

      // Advance through runs until we find the one containing the queryIndex
      while (!currentRun.done) {
        const [value, runStart, runEnd] = currentRun.value;
        if (queryIndex >= runStart && queryIndex < runEnd) {
          results[originalPos] = value;
          break;
        }
        currentRun = runIterator.next();
      }
    }

    return results;
  }

  /**
   * Checks if all boolean values within a specified range [start, end)
   * are equal to a given expectedValue.
   *
   * @param start - The starting index of the range (inclusive).
   * @param end - The ending index of the range (exclusive).
   * @param expectedValue - The boolean value to check for.
   * @returns true if every value in the range is expectedValue, false otherwise.
   *          Returns true for an empty range (start >= end).
   */
  isRangeAllValue(start: number, end: number, expectedValue: boolean): boolean {
    if (start >= end) {
      return true;
    }
    if (end > this.totalSize) {
      return false;
    }

    // Track how much of the target range we've verified
    let posCovered = start;

    for (const [runValue, runStart, runEnd] of this.iterRuns()) {
      // Find intersection of current run [runStart, runEnd) and [posCovered, end)
      const intersectStart = Math.max(runStart, posCovered);
      const intersectEnd = Math.min(runEnd, end);

      if (intersectStart < intersectEnd) {
        // There's a valid intersection - check if value matches
        if (runValue !== expectedValue) {
          return false;
        }
        // Update coverage marker
        posCovered = intersectEnd;

        // If we've covered the entire target range, we're done
        if (posCovered >= end) {
          return true;
        }
      }

      // Optimization: if current run extends beyond target, no need to check more
      if (runEnd >= end) {
        break;
      }
    }

    return posCovered >= end;
  }

  /**
   * Extends this Brle with another one by concatenating their sequences.
   *
   * @param other - The Brle to append to this one.
   */
  extend(other: Brle): void {
    if (other.isEmpty()) {
      return;
    }
    if (this.isEmpty()) {
      this.buffer = [...other.buffer];
      this.totalSize = other.totalSize;
      return;
    }

    const selfLastRunIsTrue = (this.buffer.length - 1) % 2 !== 0;
    const otherFirstRunIsTrue = other.buffer[0] === 0 && other.buffer.length > 1;

    if (selfLastRunIsTrue === otherFirstRunIsTrue) {
      // Merge the last run of self with the first run of other
      const otherFirstRunLen = otherFirstRunIsTrue
        ? other.buffer[1]
        : other.buffer[0];
      const otherSliceStart = otherFirstRunIsTrue ? 2 : 1;

      this.buffer[this.buffer.length - 1] += otherFirstRunLen;
      this.buffer.push(...other.buffer.slice(otherSliceStart));
    } else {
      // No merge needed - skip zero-length false run if present
      if (otherFirstRunIsTrue) {
        this.buffer.push(...other.buffer.slice(1));
      } else {
        this.buffer.push(...other.buffer);
      }
    }
    this.totalSize += other.totalSize;
  }

  /**
   * Core masking logic for pre-sorted, disjoint ranges.
   */
  private maskInternal(ranges: [number, number][], flag: boolean): void {
    if (ranges.length === 0 || this.totalSize === 0) {
      return;
    }

    // Collect all event points
    const events = new Set<number>();
    events.add(0);
    events.add(this.totalSize);

    for (const [start, end] of ranges) {
      const clampedStart = Math.min(start, this.totalSize);
      const clampedEnd = Math.min(end, this.totalSize);
      if (clampedStart < clampedEnd) {
        events.add(clampedStart);
        events.add(clampedEnd);
      }
    }

    for (const [, runStart, runEnd] of this.iterRuns()) {
      events.add(runStart);
      events.add(runEnd);
    }

    const eventPoints = [...events].sort((a, b) => a - b);
    const newBuffer: number[] = [];
    const runIterator = this.iterRuns();
    let rangeIndex = 0;
    let currentRun = runIterator.next();

    for (let i = 0; i < eventPoints.length - 1; i++) {
      const start = eventPoints[i];
      const end = eventPoints[i + 1];
      if (start >= end) continue;

      const midPoint = start + Math.floor((end - start) / 2);

      // Check if midPoint is in a masked range
      while (rangeIndex < ranges.length && midPoint >= ranges[rangeIndex][1]) {
        rangeIndex++;
      }
      const isMasked =
        rangeIndex < ranges.length &&
        midPoint >= ranges[rangeIndex][0] &&
        midPoint < ranges[rangeIndex][1];

      let value: boolean;
      if (isMasked) {
        value = flag;
      } else {
        // Find the original value
        while (!currentRun.done && midPoint >= currentRun.value[2]) {
          currentRun = runIterator.next();
        }
        value = currentRun.value![0];
      }

      const len = end - start;

      // Merge with previous run if same value
      const shouldMerge =
        newBuffer.length > 0 &&
        (newBuffer.length - 1) % 2 !== 0 === value;

      if (shouldMerge) {
        newBuffer[newBuffer.length - 1] += len;
      } else {
        if (newBuffer.length === 0 && value) {
          newBuffer.push(0);
        }
        newBuffer.push(len);
      }
    }

    this.buffer = newBuffer;
  }

  /**
   * Creates a clone of this Brle.
   */
  clone(): Brle {
    return new Brle([...this.buffer], this.totalSize);
  }

  /**
   * Removes a range of boolean values. The range is exclusive (`start..end`).
   * Creates a new Brle by concatenating the parts before and after the range.
   */
  removeRange(start: number, end: number): void {
    const clampedEnd = Math.min(end, this.totalSize);
    if (start >= clampedEnd) {
      return;
    }

    const head = this.slice(0, start);
    const tail = this.slice(clampedEnd, this.totalSize);

    // Merge head and tail
    if (head.totalSize === 0) {
      this.buffer = [...tail.buffer];
      this.totalSize = tail.totalSize;
    } else if (tail.totalSize === 0) {
      this.buffer = [...head.buffer];
      this.totalSize = head.totalSize;
    } else {
      // Need to properly merge - check if last run of head and first run of tail have same value
      const headLastIsTrue = (head.buffer.length - 1) % 2 !== 0;
      const tailFirstIsTrue = tail.buffer[0] === 0 && tail.buffer.length > 1;

      if (headLastIsTrue === tailFirstIsTrue) {
        // Merge the runs
        const tailFirstRunLen = tailFirstIsTrue ? tail.buffer[1] : tail.buffer[0];
        const tailSliceStart = tailFirstIsTrue ? 2 : 1;

        const newBuffer = [...head.buffer];
        newBuffer[newBuffer.length - 1] += tailFirstRunLen;
        newBuffer.push(...tail.buffer.slice(tailSliceStart));

        this.buffer = newBuffer;
      } else {
        // No merge needed
        if (tailFirstIsTrue) {
          this.buffer = [...head.buffer, ...tail.buffer.slice(1)];
        } else {
          this.buffer = [...head.buffer, ...tail.buffer];
        }
      }
      this.totalSize = head.totalSize + tail.totalSize;
    }
  }

  /**
   * Truncates the Brle to the first `newSize` elements.
   * If newSize >= totalSize, returns a clone unchanged.
   */
  truncate(newSize: number): Brle {
    if (newSize >= this.totalSize) {
      return this.clone();
    }
    return this.slice(0, newSize);
  }

  /**
   * Creates a new Brle representing a slice of the current one.
   */
  private slice(start: number, end: number): Brle {
    const clampedEnd = Math.min(end, this.totalSize);
    if (start >= clampedEnd) {
      return Brle.new(0);
    }

    const newSize = clampedEnd - start;
    const newBuffer: number[] = [];

    for (const [val, rStart, rEnd] of this.iterRuns()) {
      const sliceRStart = Math.max(rStart, start);
      const sliceREnd = Math.min(rEnd, clampedEnd);

      if (sliceRStart < sliceREnd) {
        const len = sliceREnd - sliceRStart;

        if (newBuffer.length === 0) {
          if (val) {
            newBuffer.push(0);
          }
          newBuffer.push(len);
        } else {
          const lastRunIsTrue = (newBuffer.length - 1) % 2 !== 0;
          if (lastRunIsTrue === val) {
            newBuffer[newBuffer.length - 1] += len;
          } else {
            newBuffer.push(len);
          }
        }
      }
    }

    return new Brle(newBuffer, newSize);
  }
}

/**
 * Creates a causal attention mask for the given parameters.
 * Each token can only attend to tokens at positions <= its own position.
 *
 * @param numTotalTokens - Total number of tokens in the KV cache
 * @param numInputTokens - Number of new input tokens being processed
 * @returns Array of Brle masks, one per input token
 */
export function causalMask(numTotalTokens: number, numInputTokens: number): Brle[] {
  const masks: Brle[] = [];
  const offset = numTotalTokens - numInputTokens;
  for (let i = 0; i < numInputTokens; i++) {
    masks.push(Brle.new(offset + i + 1));
  }
  return masks;
}
