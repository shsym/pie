// Model and Queue wrapper classes for inferlet-core bindings
// These classes wrap the WASM component bindings with a clean TypeScript API

import * as runtime from 'inferlet:core/runtime';
import * as apiCommon from 'inferlet:core/common';
import * as apiForward from 'inferlet:core/forward';
import * as apiAdapter from 'inferlet:adapter/common';
import * as apiZo from 'inferlet:zo/evolve';
import type {
  Model as ModelResource,
  Queue as QueueResource,
  Priority,
} from 'inferlet:core/common';
import { Tokenizer } from './tokenizer.js';
import { ForwardPass, KvPage, Resource, type Forward } from './forward.js';
import { awaitFuture } from './async-utils.js';
import { Blob } from './messaging.js';
import type { Context } from './context.js';

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


  /**
   * Begins a synchronization process for the queue (async)
   * Returns true if synchronization was successful
   */
  async synchronize(): Promise<boolean> {
    return awaitFuture(this.#inner.synchronize(), 'synchronize result was undefined');
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
    return awaitFuture(this.#inner.debugQuery(query), 'debugQuery result was undefined');
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

  // Adapter management (Resource.Adapter = 2)

  /**
   * Allocate a single adapter pointer
   */
  allocateAdapter(): number {
    const ptrs = this.allocateResources(Resource.Adapter, 1);
    return ptrs[0];
  }

  /**
   * Deallocate an adapter pointer
   */
  deallocateAdapter(ptr: number): void {
    this.deallocateResources(Resource.Adapter, [ptr]);
  }

  /**
   * Export an adapter for later import
   */
  exportAdapter(ptr: number, name: string): void {
    this.exportResources(Resource.Adapter, [ptr], name);
  }

  /**
   * Import an adapter by name
   */
  importAdapter(name: string): number {
    const ptrs = this.importResources(Resource.Adapter, name);
    return ptrs[0];
  }

  /**
   * Get all exported adapter names
   */
  getAllExportedAdapters(): string[] {
    return this.getAllExportedResources(Resource.Adapter).map(([name]) => name);
  }

  /**
   * Release an exported adapter by name
   */
  releaseExportedAdapter(name: string): void {
    this.releaseExportedResources(Resource.Adapter, name);
  }

  /**
   * Upload adapter weights
   */
  uploadAdapter(adapterPtr: number, name: string, blob: Blob): void {
    apiAdapter.uploadAdapter(this.#inner, adapterPtr, name, blob.getInner());
  }

  /**
   * Download adapter weights (async)
   */
  async downloadAdapter(adapterPtr: number, name: string): Promise<Blob> {
    const future = apiAdapter.downloadAdapter(this.#inner, adapterPtr, name);
    const blobInner = awaitFuture(future, 'downloadAdapter result was undefined');
    return Blob.fromWit(blobInner);
  }

  // Zero-Order Evolution methods

  /**
   * Initialize an adapter for zero-order evolution.
   * @param adapterPtr - Pointer to the adapter resource
   * @param rank - LoRA rank
   * @param alpha - LoRA alpha scaling factor
   * @param populationSize - Number of candidates per generation
   * @param muFraction - Fraction of top performers to keep
   * @param initialSigma - Initial perturbation standard deviation
   */
  initializeAdapter(
    adapterPtr: number,
    rank: number,
    alpha: number,
    populationSize: number,
    muFraction: number,
    initialSigma: number
  ): void {
    apiZo.initializeAdapter(
      this.#inner,
      adapterPtr,
      rank,
      alpha,
      populationSize,
      muFraction,
      initialSigma
    );
  }

  /**
   * Update adapter weights based on evolution scores.
   * @param adapterPtr - Pointer to the adapter resource
   * @param scores - Fitness scores for each candidate
   * @param seeds - Random seeds used for each candidate's perturbation
   * @param maxSigma - Maximum allowed sigma value
   */
  updateAdapter(
    adapterPtr: number,
    scores: number[],
    seeds: bigint[],
    maxSigma: number
  ): void {
    apiZo.updateAdapter(
      this.#inner,
      adapterPtr,
      new Float32Array(scores),
      new BigInt64Array(seeds),
      maxSigma
    );
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
   * Get all exported KV pages (method form, required by Forward interface)
   */
  getAllExportedKvPages(): [string, number][] {
    return this.getAllExportedResources(Resource.KvPage);
  }

  /**
   * All exported KV pages (getter form)
   */
  get allExportedKvPages(): [string, number][] {
    return this.getAllExportedKvPages();
  }

  /**
   * Get all exported embeddings (method form, required by Forward interface)
   */
  getAllExportedEmbeds(): [string, number][] {
    return this.getAllExportedResources(Resource.Embed);
  }

  /**
   * All exported embeddings (getter form)
   */
  get allExportedEmbeds(): [string, number][] {
    return this.getAllExportedEmbeds();
  }

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

  /**
   * Create a new context for this model.
   * This is a convenience method equivalent to `new Context(model)`.
   *
   * Note: Uses late binding to resolve circular dependency with Context module.
   */
  createContext(): Context {
    return _createContext(this);
  }

}

// Late binding for Context to avoid circular dependency at module load time.
// The actual Context class is bound when createContextBinding() is called from context.ts.
let _ContextClass: (new (model: Model) => Context) | null = null;

function _createContext(model: Model): Context {
  if (!_ContextClass) {
    throw new Error('Context class not bound. Ensure context.js is imported.');
  }
  return new _ContextClass(model);
}

/**
 * Binds the Context class for late binding.
 * Called from context.ts to resolve circular dependency.
 * @internal
 */
export function _bindContextClass(ContextClass: new (model: Model) => Context): void {
  _ContextClass = ContextClass;
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
