// Model and Queue wrapper classes for inferlet-core bindings
// These classes wrap the WASM component bindings with a clean TypeScript API

import * as runtime from 'inferlet:core/runtime';
import * as apiCommon from 'inferlet:core/common';
import * as apiForward from 'inferlet:core/forward';
import type {
  Model as ModelResource,
  Queue as QueueResource,
  Priority,
  SynchronizationResult,
  DebugQueryResult,
} from 'inferlet:core/common';
import type { Pollable } from 'wasi:io/poll';
import { Tokenizer } from './tokenizer.js';
import { ForwardPass, KvPage, Resource, type Forward } from './forward.js';

/**
 * Represents a command queue for a specific model instance.
 * Queues manage the execution of commands and resource allocation.
 * Implements the Forward interface for creating forward passes and managing resources.
 */
export class Queue implements Forward {
  readonly #inner: QueueResource;
  readonly #serviceId: number;

  constructor(inner: QueueResource, serviceId: number) {
    this.#inner = inner;
    this.#serviceId = serviceId;
  }

  /**
   * The raw inner queue resource (for internal use)
   */
  get inner(): QueueResource {
    return this.#inner;
  }

  /**
   * The service ID for the queue
   */
  get serviceId(): number {
    return this.#serviceId;
  }

  // Deprecated methods for backward compatibility
  /** @deprecated Use `inner` getter instead */
  getInner(): QueueResource { return this.#inner; }
  /** @deprecated Use `serviceId` getter instead */
  getServiceId(): number { return this.#serviceId; }

  /**
   * Begins a synchronization process for the queue (async)
   * Returns true if synchronization was successful
   */
  async synchronize(): Promise<boolean> {
    const result: SynchronizationResult = this.#inner.synchronize();
    const pollable: Pollable = result.pollable();

    // Block until the result is ready
    pollable.block();

    // Get the result
    const value = result.get();

    if (value === undefined) {
      throw new Error('synchronize result was undefined after pollable was ready');
    }

    return value;
  }

  /**
   * Change the queue's priority
   */
  setPriority(priority: Priority): void {
    this.#inner.setPriority(priority);
  }

  /**
   * Executes a debug command on the queue and returns the result (async)
   */
  async debugQuery(query: string): Promise<string> {
    const result: DebugQueryResult = this.#inner.debugQuery(query);
    const pollable: Pollable = result.pollable();

    // Block until the result is ready
    pollable.block();

    // Get the result
    const value = result.get();

    if (value === undefined) {
      throw new Error('debugQuery result was undefined after pollable was ready');
    }

    return value;
  }

  // ============================================
  // Forward interface implementation
  // ============================================

  /**
   * Allocate resources of the specified type
   */
  allocateResources(resource: Resource, count: number): number[] {
    return [...apiCommon.allocateResources(this.#inner, resource, count)];
  }

  /**
   * Deallocate resources of the specified type
   */
  deallocateResources(resource: Resource, ptrs: number[]): void {
    apiCommon.deallocateResources(this.#inner, resource, new Uint32Array(ptrs));
  }

  /**
   * Export resources with a name for later import
   */
  exportResources(resource: Resource, ptrs: number[], name: string): void {
    apiCommon.exportResources(this.#inner, resource, new Uint32Array(ptrs), name);
  }

  /**
   * Import resources by name
   */
  importResources(resource: Resource, name: string): number[] {
    return [...apiCommon.importResources(this.#inner, resource, name)];
  }

  /**
   * Get all exported resources of a type
   */
  getAllExportedResources(resource: Resource): [string, number][] {
    return apiCommon.getAllExportedResources(this.#inner, resource).map(([name, count]) => [name, count]);
  }

  /**
   * Release exported resources by name
   */
  releaseExportedResources(resource: Resource, name: string): void {
    apiCommon.releaseExportedResources(this.#inner, resource, name);
  }

  // KvPage management with smart pointers

  /**
   * Allocate a new KV page with automatic resource management
   */
  newKvPage(): KvPage {
    const ptr = this.allocateKvPagePtr();
    return new KvPage(this, ptr);
  }

  /**
   * Allocate multiple KV pages with automatic resource management
   */
  newKvPages(count: number): KvPage[] {
    return this.allocateKvPagePtrs(count).map((ptr) => new KvPage(this, ptr));
  }

  // KvPage raw pointer management

  /**
   * Allocate a single KV page pointer
   */
  allocateKvPagePtr(): number {
    const ptrs = this.allocateResources(Resource.KvPage, 1);
    return ptrs[0];
  }

  /**
   * Allocate multiple KV page pointers
   */
  allocateKvPagePtrs(count: number): number[] {
    return this.allocateResources(Resource.KvPage, count);
  }

  /**
   * Deallocate a single KV page pointer
   */
  deallocateKvPagePtr(ptr: number): void {
    this.deallocateResources(Resource.KvPage, [ptr]);
  }

  /**
   * Deallocate multiple KV page pointers
   */
  deallocateKvPagePtrs(ptrs: number[]): void {
    this.deallocateResources(Resource.KvPage, ptrs);
  }

  // KvPage export/import

  /**
   * Export KV pages for later import
   */
  exportKvPages(kvPages: KvPage[], name: string): void {
    const ptrs = kvPages.map((kv) => kv.ptr);
    this.exportResources(Resource.KvPage, ptrs, name);
  }

  /**
   * Import KV pages by name
   */
  importKvPages(name: string): KvPage[] {
    const ptrs = this.importResources(Resource.KvPage, name);
    return ptrs.map((ptr) => new KvPage(this, ptr));
  }

  /**
   * Export KV page pointers for later import
   */
  exportKvPagePtrs(ptrs: number[], name: string): void {
    this.exportResources(Resource.KvPage, ptrs, name);
  }

  /**
   * Import KV page pointers by name
   */
  importKvPagePtrs(name: string): number[] {
    return this.importResources(Resource.KvPage, name);
  }

  /**
   * Release exported KV pages by name
   */
  releaseExportedKvPages(name: string): void {
    this.releaseExportedResources(Resource.KvPage, name);
  }

  // Embedding pointer management

  /**
   * Allocate a single embedding pointer
   */
  allocateEmbedPtr(): number {
    const ptrs = this.allocateResources(Resource.Embed, 1);
    return ptrs[0];
  }

  /**
   * Allocate multiple embedding pointers
   */
  allocateEmbedPtrs(count: number): number[] {
    return this.allocateResources(Resource.Embed, count);
  }

  /**
   * Deallocate a single embedding pointer
   */
  deallocateEmbedPtr(ptr: number): void {
    this.deallocateResources(Resource.Embed, [ptr]);
  }

  /**
   * Deallocate multiple embedding pointers
   */
  deallocateEmbedPtrs(ptrs: number[]): void {
    this.deallocateResources(Resource.Embed, ptrs);
  }

  /**
   * Export embedding pointers for later import
   */
  exportEmbedPtrs(ptrs: number[], name: string): void {
    this.exportResources(Resource.Embed, ptrs, name);
  }

  /**
   * Import embedding pointers by name
   */
  importEmbedPtrs(name: string): number[] {
    return this.importResources(Resource.Embed, name);
  }

  /**
   * Release exported embeddings by name
   */
  releaseExportedEmbeds(name: string): void {
    this.releaseExportedResources(Resource.Embed, name);
  }

  // ForwardPass creation

  /**
   * Create a new forward pass for executing model inference
   */
  createForwardPass(): ForwardPass {
    const inner = apiForward.createForwardPass(this.#inner);
    return new ForwardPass(inner);
  }

  /**
   * All exported KV pages
   */
  get allExportedKvPages(): [string, number][] {
    return this.getAllExportedResources(Resource.KvPage);
  }

  /**
   * All exported embeddings
   */
  get allExportedEmbeds(): [string, number][] {
    return this.getAllExportedResources(Resource.Embed);
  }

  // Deprecated methods
  /** @deprecated Use `allExportedKvPages` getter instead */
  getAllExportedKvPages(): [string, number][] { return this.allExportedKvPages; }
  /** @deprecated Use `allExportedEmbeds` getter instead */
  getAllExportedEmbeds(): [string, number][] { return this.allExportedEmbeds; }
}

/**
 * Represents a specific model instance, providing access to its metadata and functionality.
 */
export class Model {
  private readonly inner: ModelResource;

  // Lazy caches
  #tokenizerCache: Tokenizer | null = null;
  #eosTokensCache: number[][] | null = null;

  constructor(inner: ModelResource) {
    this.inner = inner;
  }

  /**
   * The model's name (e.g. "llama-3.1-8b-instruct")
   */
  get name(): string {
    return this.inner.getName();
  }

  /**
   * The full set of model traits
   */
  get traits(): string[] {
    return this.inner.getTraits();
  }

  /**
   * Checks if the model has all the specified traits
   */
  hasTraits(requiredTraits: string[]): boolean {
    const availableTraits = new Set(this.traits);
    return requiredTraits.every(trait => availableTraits.has(trait));
  }

  /**
   * A human-readable description of the model
   */
  get description(): string {
    return this.inner.getDescription();
  }

  /**
   * The prompt formatting template in Jinja format
   */
  get promptTemplate(): string {
    return this.inner.getPromptTemplate();
  }

  /**
   * The stop tokens for the model (as strings)
   */
  get stopTokens(): string[] {
    return this.inner.getStopTokens();
  }

  /**
   * The service ID for the model
   */
  get serviceId(): number {
    return this.inner.getServiceId();
  }

  /**
   * The size of a KV page for this model
   */
  get kvPageSize(): number {
    return this.inner.getKvPageSize();
  }

  /**
   * The tokenizer for this model (lazy cached)
   */
  get tokenizer(): Tokenizer {
    if (!this.#tokenizerCache) {
      this.#tokenizerCache = new Tokenizer(this.inner);
    }
    return this.#tokenizerCache;
  }

  /**
   * The EOS (end-of-sequence) tokens as tokenized arrays (lazy cached)
   */
  get eosTokens(): number[][] {
    if (!this.#eosTokensCache) {
      this.#eosTokensCache = this.stopTokens.map(stopToken =>
        [...this.tokenizer.tokenize(stopToken)]
      );
    }
    return this.#eosTokensCache;
  }

  /**
   * Create a new command queue for this model
   */
  createQueue(): Queue {
    const queueResource = this.inner.createQueue();
    return new Queue(queueResource, this.serviceId);
  }

  // Deprecated methods for backward compatibility
  /** @deprecated Use `name` getter instead */
  getName(): string { return this.name; }
  /** @deprecated Use `traits` getter instead */
  getTraits(): string[] { return this.traits; }
  /** @deprecated Use `description` getter instead */
  getDescription(): string { return this.description; }
  /** @deprecated Use `promptTemplate` getter instead */
  getPromptTemplate(): string { return this.promptTemplate; }
  /** @deprecated Use `stopTokens` getter instead */
  getStopTokens(): string[] { return this.stopTokens; }
  /** @deprecated Use `serviceId` getter instead */
  getServiceId(): number { return this.serviceId; }
  /** @deprecated Use `kvPageSize` getter instead */
  getKvPageSize(): number { return this.kvPageSize; }
  /** @deprecated Use `tokenizer` getter instead */
  getTokenizer(): Tokenizer { return this.tokenizer; }
}

/**
 * Retrieve a model by its name
 * Returns undefined if no model with the specified name is found
 */
export function getModel(name: string): Model | undefined {
  const modelResource = runtime.getModel(name);
  if (modelResource === undefined) {
    return undefined;
  }
  return new Model(modelResource);
}

/**
 * Get a list of all available model names
 */
export function getAllModels(): string[] {
  return runtime.getAllModels();
}

/**
 * Get the first available model automatically
 * Throws an error if no models are available
 */
export function getAutoModel(): Model {
  const models = getAllModels();
  if (models.length === 0) {
    throw new Error('No models available');
  }
  const model = getModel(models[0]);
  if (model === undefined) {
    throw new Error(`Model ${models[0]} not found`);
  }
  return model;
}

/**
 * Get names of models that have all specified traits (e.g. "input_text", "tokenize")
 */
export function getAllModelsWithTraits(traits: string[]): string[] {
  return runtime.getAllModelsWithTraits(traits);
}
