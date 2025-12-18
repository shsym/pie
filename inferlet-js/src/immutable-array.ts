/**
 * Copy-on-write immutable array wrapper.
 * Shares underlying array until mutation is required.
 */
export class ImmutableArray<T> {
  private data: T[];
  private start: number;
  private end: number;
  private owned: boolean;

  private constructor(data: T[], start: number, end: number, owned: boolean) {
    this.data = data;
    this.start = start;
    this.end = end;
    this.owned = owned;
  }

  /**
   * Create from an existing array (takes ownership)
   */
  static from<T>(arr: T[]): ImmutableArray<T> {
    return new ImmutableArray(arr, 0, arr.length, true);
  }

  /**
   * Create an empty array
   */
  static empty<T>(): ImmutableArray<T> {
    return new ImmutableArray([], 0, 0, true);
  }

  get length(): number {
    return this.end - this.start;
  }

  /**
   * Get element at index (supports negative indexing)
   */
  get(index: number): T | undefined {
    if (index < 0) {
      index = this.length + index;
    }
    if (index < 0 || index >= this.length) {
      return undefined;
    }
    return this.data[this.start + index];
  }

  /**
   * Create a shallow fork that shares the underlying array
   */
  fork(): ImmutableArray<T> {
    return new ImmutableArray(this.data, this.start, this.end, false);
  }

  /**
   * Return a new array with element pushed (copy-on-write)
   */
  push(item: T): ImmutableArray<T> {
    const newData = this.toArray();
    newData.push(item);
    return new ImmutableArray(newData, 0, newData.length, true);
  }

  /**
   * Return a new array with elements pushed (copy-on-write)
   */
  pushAll(items: T[]): ImmutableArray<T> {
    if (items.length === 0) return this;
    const newData = this.toArray();
    newData.push(...items);
    return new ImmutableArray(newData, 0, newData.length, true);
  }

  /**
   * Slice without copying underlying array
   */
  slice(start?: number, end?: number): ImmutableArray<T> {
    const len = this.length;
    const s = start === undefined ? 0 : start < 0 ? Math.max(0, len + start) : Math.min(start, len);
    const e = end === undefined ? len : end < 0 ? Math.max(0, len + end) : Math.min(end, len);

    return new ImmutableArray(
      this.data,
      this.start + s,
      this.start + Math.max(s, e),
      false
    );
  }

  /**
   * Concat with regular array
   */
  concat(other: T[]): ImmutableArray<T> {
    if (other.length === 0) return this;
    const newData = this.toArray();
    newData.push(...other);
    return new ImmutableArray(newData, 0, newData.length, true);
  }

  /**
   * Convert to regular array (creates copy if not owned)
   */
  toArray(): T[] {
    if (this.owned && this.start === 0 && this.end === this.data.length) {
      // We own the full array, return a copy to maintain immutability
      return [...this.data];
    }
    return this.data.slice(this.start, this.end);
  }

  /**
   * Iterate over elements
   */
  *[Symbol.iterator](): Iterator<T> {
    for (let i = this.start; i < this.end; i++) {
      yield this.data[i];
    }
  }

  /**
   * Map to regular array
   */
  map<U>(fn: (value: T, index: number) => U): U[] {
    const result: U[] = [];
    let idx = 0;
    for (let i = this.start; i < this.end; i++) {
      result.push(fn(this.data[i], idx++));
    }
    return result;
  }
}
