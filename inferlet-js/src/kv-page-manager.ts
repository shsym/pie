import type { Queue } from './model.js';
import type { KvPage } from './forward.js';

/**
 * Manages KV cache page allocation and lifecycle.
 * Extracted from Context to reduce class size.
 */
export class KvPageManager {
  private queue: Queue;
  private pageSize: number;
  private pages: KvPage[] = [];
  private _lastPageLen: number = 0;

  constructor(queue: Queue, pageSize: number) {
    this.queue = queue;
    this.pageSize = pageSize;
  }

  get pageCount(): number {
    return this.pages.length;
  }

  get lastPageLen(): number {
    return this._lastPageLen;
  }

  get totalTokens(): number {
    if (this.pages.length === 0) return this._lastPageLen;
    return (this.pages.length - 1) * this.pageSize + this._lastPageLen;
  }

  get ptrs(): number[] {
    return this.pages.map(p => p.ptr);
  }

  get allPages(): KvPage[] {
    return this.pages;
  }

  /**
   * Grow the KV cache to accommodate more tokens
   */
  grow(numTokens: number): void {
    this.adjust(numTokens);
  }

  /**
   * Shrink the KV cache
   */
  shrink(numTokens: number): void {
    this.adjust(-numTokens);
  }

  /**
   * Fork for copy-on-write sharing
   */
  fork(): KvPageManager {
    const forked = new KvPageManager(this.queue, this.pageSize);
    forked.pages = [...this.pages];
    forked._lastPageLen = this._lastPageLen;

    // Increment ref count for shared pages
    for (const page of forked.pages) {
      page.ref();
    }

    return forked;
  }

  /**
   * Fork with partial page handling (hard case)
   */
  forkPartial(): { manager: KvPageManager; droppedTokenCount: number } {
    const keptPageCount = Math.max(0, this.pages.length - 1);
    const keptTokens = keptPageCount * this.pageSize;
    const droppedTokenCount = this.totalTokens - keptTokens;

    const forked = new KvPageManager(this.queue, this.pageSize);
    forked.pages = this.pages.slice(0, keptPageCount);
    forked._lastPageLen = keptPageCount > 0 ? this.pageSize : 0;

    // Increment ref count for shared pages
    for (const page of forked.pages) {
      page.ref();
    }

    return { manager: forked, droppedTokenCount };
  }

  /**
   * Release all pages
   */
  release(): void {
    for (const page of this.pages) {
      page.release();
    }
    this.pages = [];
    this._lastPageLen = 0;
  }

  /**
   * Adopt pages from another manager (for beam search winner)
   */
  adopt(other: KvPageManager): void {
    this.release();
    this.pages = [...other.pages];
    this._lastPageLen = other._lastPageLen;

    // Increment ref count since we're now an owner
    for (const page of this.pages) {
      page.ref();
    }
  }

  private adjust(numTokens: number): void {
    if (numTokens === 0) return;

    const currentTokens = this.totalTokens;
    const newTotalTokens = currentTokens + numTokens;

    if (newTotalTokens < 0) {
      throw new Error('Token count adjustment resulted in underflow');
    }

    const currentPages = this.pages.length;
    const requiredPages = Math.ceil(newTotalTokens / this.pageSize);

    if (requiredPages > currentPages) {
      // Grow: Allocate new pages
      const newPagesNeeded = requiredPages - currentPages;
      const newKvPages = this.queue.newKvPages(newPagesNeeded);
      this.pages.push(...newKvPages);
    } else if (requiredPages < currentPages) {
      // Shrink: Release excess pages
      const pagesToRelease = this.pages.splice(requiredPages);
      for (const page of pagesToRelease) {
        page.release();
      }
    }

    // Update the length of the last page
    const lastPageLen = newTotalTokens % this.pageSize;
    this._lastPageLen = lastPageLen === 0 && newTotalTokens > 0
      ? this.pageSize
      : lastPageLen;
  }
}
