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
   * Fork for copy-on-write sharing.
   *
   * IMPORTANT: Only shares FULL pages. Partial pages are dropped because:
   * - Shared pages point to the same memory
   * - If two contexts write to the same position, they corrupt each other
   * - Full pages are read-only (new writes go to new pages)
   * - Partial pages would be written to, so must be unique per context
   */
  fork(): { manager: KvPageManager; droppedTokenCount: number } {
    // Always use forkPartial behavior to ensure no writable pages are shared.
    // This is critical for beam search where multiple contexts write independently.
    return this.forkPartial();
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

  /**
   * Import pages from external source (e.g., cache restore)
   */
  importPages(pages: KvPage[], lastPageLen: number): void {
    this.pages = pages;
    this._lastPageLen = lastPageLen;
  }

  /**
   * Remove a page at the specified index.
   * Used by dropMaskedKvPages to remove fully-masked pages.
   *
   * @param index - The index of the page to remove
   */
  removePageAt(index: number): void {
    if (index < 0 || index >= this.pages.length) {
      throw new Error(`Page index ${index} out of bounds`);
    }
    const [removed] = this.pages.splice(index, 1);
    removed.release();

    // Recalculate last page length
    const newTotalTokens = this.totalTokens;
    const lastPageLen = newTotalTokens % this.pageSize;
    this._lastPageLen =
      lastPageLen === 0 && newTotalTokens > 0 ? this.pageSize : lastPageLen;
  }

  /**
   * Recalculate last page length based on current page count and total tokens.
   * Used after external modifications to pages array.
   *
   * @param totalTokens - The total number of committed tokens
   */
  recalculateLastPageLen(totalTokens: number): void {
    const lastPageLen = totalTokens % this.pageSize;
    this._lastPageLen =
      lastPageLen === 0 && totalTokens > 0 ? this.pageSize : lastPageLen;
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
