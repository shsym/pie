// ForwardPass, KvPage, and Distribution types for neural network forward passes.
// Mirrors the Rust forward.rs from inferlet/src/forward.rs

import type { Queue } from './model.js';
import { Brle } from './brle.js';

// WIT bindings - these will be available at runtime through componentize-js
import type { ForwardPass as WitForwardPass } from 'inferlet:core/forward';
import * as forward from 'inferlet:core/forward';
import * as apiAdapter from 'inferlet:adapter/common';
import * as apiZo from 'inferlet:zo/evolve';

/**
 * Represents a probability distribution over a set of tokens.
 */
export interface Distribution {
  /** Token IDs */
  ids: number[];
  /** Probabilities corresponding to the token IDs */
  probs: number[];
}

/**
 * Result of executing a forward pass.
 */
export interface ForwardPassResult {
  /** Optional distributions for requested output indices */
  distributions?: Distribution[];
  /** Optional sampled tokens for requested output indices */
  tokens?: number[];
}

// Enable to trace KvPage lifecycle for debugging
const DEBUG_KVPAGE = false;
let kvPageIdCounter = 0;

/**
 * Represents a KV cache page with automatic resource management.
 * When the last reference is dropped, the page is deallocated.
 */
export class KvPage {
  private _queue: Queue;
  private _ptr: number;
  private _released: boolean = false;
  private _refCount: number = 1;
  private _debugId: number;

  constructor(queue: Queue, ptr: number) {
    this._queue = queue;
    this._ptr = ptr;
    this._debugId = kvPageIdCounter++;
    if (DEBUG_KVPAGE) {
      console.log(`[KvPage] CREATE id=${this._debugId} ptr=${this._ptr} refCount=${this._refCount}`);
    }
  }

  /**
   * Returns the raw pointer to this KV page.
   */
  get ptr(): number {
    return this._ptr;
  }

  /**
   * Returns the current reference count (for debugging).
   */
  get refCount(): number {
    return this._refCount;
  }

  /**
   * Returns whether this page has been released.
   */
  get isReleased(): boolean {
    return this._released;
  }

  /**
   * Increment the reference count.
   * Call this when sharing this page with another Context.
   */
  ref(): void {
    if (this._released) {
      console.error(`[KvPage] ERROR: ref() on released page id=${this._debugId} ptr=${this._ptr}`);
      throw new Error(`Cannot ref a released KvPage (ptr=${this._ptr})`);
    }
    this._refCount++;
    if (DEBUG_KVPAGE) {
      console.log(`[KvPage] REF id=${this._debugId} ptr=${this._ptr} refCount=${this._refCount}`);
    }
  }

  /**
   * Decrement the reference count and release if it reaches zero.
   * Returns true if the page was actually deallocated.
   */
  unref(): boolean {
    if (this._released) {
      if (DEBUG_KVPAGE) {
        console.log(`[KvPage] SKIP unref (already released) id=${this._debugId} ptr=${this._ptr}`);
      }
      return false;
    }
    this._refCount--;
    if (DEBUG_KVPAGE) {
      console.log(`[KvPage] UNREF id=${this._debugId} ptr=${this._ptr} refCount=${this._refCount}`);
    }
    if (this._refCount <= 0) {
      if (DEBUG_KVPAGE) {
        console.log(`[KvPage] DEALLOCATE id=${this._debugId} ptr=${this._ptr}`);
      }
      this._queue.deallocateKvPagePtr(this._ptr);
      this._released = true;
      return true;
    }
    return false;
  }

  /**
   * Explicitly release this KV page (decrements ref count).
   * Called automatically when the page is no longer needed.
   */
  release(): void {
    this.unref();
  }
}

/**
 * ForwardPass represents a batch of forward pass operations.
 * Configure inputs, outputs, attention masks, and KV cache before executing.
 */
export class ForwardPass {
  private inner: WitForwardPass;

  constructor(inner: WitForwardPass) {
    this.inner = inner;
  }

  /**
   * Execute the forward pass asynchronously.
   */
  async execute(): Promise<ForwardPassResult> {
    const future = this.inner.execute();
    if (!future) {
      return { distributions: undefined, tokens: undefined };
    }

    // Wait for the async operation to complete
    const pollable = future.pollable();
    // Use pollable.block() which is the WASI way to wait
    pollable.block();

    // Get results
    const distributions: Distribution[] = [];
    const rawDistributions = future.getDistributions();
    if (rawDistributions) {
      for (const [ids, probs] of rawDistributions) {
        distributions.push({ ids: [...ids], probs: [...probs] });
      }
    }

    const tokens = future.getTokens();

    return {
      distributions: distributions.length > 0 ? distributions : undefined,
      tokens: tokens ? [...tokens] : undefined,
    };
  }

  /**
   * Set embedding pointers as input.
   */
  inputEmbedPtrs(embedPtrs: number[], positions: number[]): void {
    forward.inputEmbeddings(
      this.inner,
      new Uint32Array(embedPtrs),
      new Uint32Array(positions)
    );
  }

  /**
   * Set token IDs as input.
   */
  inputTokens(inputTokens: number[], positions: number[]): void {
    forward.inputTokens(
      this.inner,
      new Uint32Array(inputTokens),
      new Uint32Array(positions)
    );
  }

  /**
   * Set embedding pointers for output capture.
   */
  outputEmbedPtrs(embedPtrs: number[], indices: number[]): void {
    forward.outputEmbeddings(
      this.inner,
      new Uint32Array(embedPtrs),
      new Uint32Array(indices)
    );
  }

  /**
   * Request probability distributions at specified indices.
   */
  outputDistributions(indices: number[], temperature: number, topK?: number): void {
    forward.outputDistributions(this.inner, new Uint32Array(indices), temperature, topK);
  }

  /**
   * Request sampled tokens at specified indices (multinomial sampling).
   */
  outputTokens(indices: number[], temperature: number): void {
    forward.outputTokens(this.inner, new Uint32Array(indices), temperature);
  }

  /**
   * Request sampled tokens using top-p (nucleus) sampling.
   */
  outputTokensTopP(indices: number[], temperature: number, topP: number): void {
    forward.outputTokensTopP(this.inner, new Uint32Array(indices), temperature, topP);
  }

  /**
   * Request sampled tokens using top-k sampling.
   */
  outputTokensTopK(indices: number[], temperature: number, topK: number): void {
    forward.outputTokensTopK(this.inner, new Uint32Array(indices), temperature, topK);
  }

  /**
   * Request sampled tokens using min-p sampling.
   */
  outputTokensMinP(indices: number[], temperature: number, minP: number): void {
    forward.outputTokensMinP(this.inner, new Uint32Array(indices), temperature, minP);
  }

  /**
   * Request sampled tokens using combined top-k and top-p sampling.
   */
  outputTokensTopKTopP(
    indices: number[],
    temperature: number,
    topK: number,
    topP: number
  ): void {
    forward.outputTokensTopKTopP(this.inner, new Uint32Array(indices), temperature, topK, topP);
  }

  /**
   * Set the attention mask for the forward pass.
   * Each element is a Brle or raw buffer representing which positions are visible.
   */
  attentionMask(mask: Brle[] | number[][]): void {
    // Convert Brle array to Uint32Array[] as expected by the WIT binding
    const rawMask = mask.map((m) => {
      const buf = m instanceof Brle ? m.buffer : m;
      return new Uint32Array(buf);
    });
    forward.attentionMask(this.inner, rawMask);
  }

  /**
   * Set the KV cache for the forward pass.
   */
  kvCache(kvPages: KvPage[], lastKvPageLen: number): void {
    const ptrs = kvPages.map((kv) => kv.ptr);
    forward.kvCache(this.inner, new Uint32Array(ptrs), lastKvPageLen);
  }

  /**
   * Set the KV cache using raw pointers.
   */
  kvCachePtrs(kvPagePtrs: number[], lastKvPageLen: number): void {
    forward.kvCache(this.inner, new Uint32Array(kvPagePtrs), lastKvPageLen);
  }

  /**
   * Set the adapter for this forward pass (LoRA inference).
   * @param adapterPtr - Pointer to the adapter resource
   */
  setAdapter(adapterPtr: number): void {
    apiAdapter.setAdapter(this.inner, adapterPtr);
  }

  /**
   * Set the random seed for adapter perturbation (ZO optimization).
   * @param seed - Random seed for perturbation
   */
  setAdapterSeed(seed: number): void {
    apiZo.setAdapterSeed(this.inner, BigInt(seed));
  }
}

/**
 * Resource type enumeration for allocation/deallocation.
 */
export enum Resource {
  KvPage = 0,
  Embed = 1,
  Adapter = 2,
}

/**
 * Interface for forward pass operations.
 * Implemented by Queue.
 */
export interface Forward {
  // KvPage management with smart pointers
  newKvPage(): KvPage;
  newKvPages(count: number): KvPage[];

  // KvPage raw pointer management
  allocateKvPagePtr(): number;
  allocateKvPagePtrs(count: number): number[];
  deallocateKvPagePtr(ptr: number): void;
  deallocateKvPagePtrs(ptrs: number[]): void;

  // KvPage export/import
  exportKvPages(kvPages: KvPage[], name: string): void;
  importKvPages(name: string): KvPage[];
  exportKvPagePtrs(ptrs: number[], name: string): void;
  importKvPagePtrs(name: string): number[];
  getAllExportedKvPages(): [string, number][];
  releaseExportedKvPages(name: string): void;

  // Embedding pointer management
  allocateEmbedPtr(): number;
  allocateEmbedPtrs(count: number): number[];
  deallocateEmbedPtr(ptr: number): void;
  deallocateEmbedPtrs(ptrs: number[]): void;
  exportEmbedPtrs(ptrs: number[], name: string): void;
  importEmbedPtrs(name: string): number[];
  getAllExportedEmbeds(): [string, number][];
  releaseExportedEmbeds(name: string): void;

  // ForwardPass creation
  createForwardPass(): ForwardPass;
}

// Re-export causalMask from brle
export { causalMask } from './brle.js';
