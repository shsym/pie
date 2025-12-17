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
  private readonly inner: QueueResource;
  private readonly serviceId: number;

  constructor(inner: QueueResource, serviceId: number) {
    this.inner = inner;
    this.serviceId = serviceId;
  }

  /**
   * Gets the raw inner queue resource (for internal use)
   */
  getInner(): QueueResource {
    return this.inner;
  }

  /**
   * Gets the service ID for the queue
   */
  getServiceId(): number {
    return this.serviceId;
  }

  /**
   * Begins a synchronization process for the queue (async)
   * Returns true if synchronization was successful
   */
  async synchronize(): Promise<boolean> {
    const result: SynchronizationResult = this.inner.synchronize();
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
    this.inner.setPriority(priority);
  }

  /**
   * Executes a debug command on the queue and returns the result (async)
   */
  async debugQuery(query: string): Promise<string> {
    const result: DebugQueryResult = this.inner.debugQuery(query);
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
    return [...apiCommon.allocateResources(this.inner, resource, count)];
  }

  /**
   * Deallocate resources of the specified type
   */
  deallocateResources(resource: Resource, ptrs: number[]): void {
    apiCommon.deallocateResources(this.inner, resource, ptrs);
  }

  /**
   * Export resources with a name for later import
   */
  exportResources(resource: Resource, ptrs: number[], name: string): void {
    apiCommon.exportResources(this.inner, resource, ptrs, name);
  }

  /**
   * Import resources by name
   */
  importResources(resource: Resource, name: string): number[] {
    return [...apiCommon.importResources(this.inner, resource, name)];
  }

  /**
   * Get all exported resources of a type
   */
  getAllExportedResources(resource: Resource): [string, number][] {
    return apiCommon.getAllExportedResources(this.inner, resource).map(([name, count]) => [name, count]);
  }

  /**
   * Release exported resources by name
   */
  releaseExportedResources(resource: Resource, name: string): void {
    apiCommon.releaseExportedResources(this.inner, resource, name);
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
   * Get all exported KV pages
   */
  getAllExportedKvPages(): [string, number][] {
    return this.getAllExportedResources(Resource.KvPage);
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
   * Get all exported embeddings
   */
  getAllExportedEmbeds(): [string, number][] {
    return this.getAllExportedResources(Resource.Embed);
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
    const inner = apiForward.createForwardPass(this.inner);
    return new ForwardPass(inner);
  }
}

/**
 * Represents a specific model instance, providing access to its metadata and functionality.
 */
export class Model {
  private readonly inner: ModelResource;

  constructor(inner: ModelResource) {
    this.inner = inner;
  }

  /**
   * Returns the model's name (e.g. "llama-3.1-8b-instruct")
   */
  getName(): string {
    return this.inner.getName();
  }

  /**
   * Returns the full set of model traits
   */
  getTraits(): string[] {
    return this.inner.getTraits();
  }

  /**
   * Checks if the model has all the specified traits
   */
  hasTraits(requiredTraits: string[]): boolean {
    const availableTraits = new Set(this.getTraits());
    return requiredTraits.every(trait => availableTraits.has(trait));
  }

  /**
   * Returns a human-readable description of the model
   */
  getDescription(): string {
    return this.inner.getDescription();
  }

  /**
   * Returns the prompt formatting template in Tera format
   */
  getPromptTemplate(): string {
    return this.inner.getPromptTemplate();
  }

  /**
   * Returns the stop tokens for the model
   */
  getStopTokens(): string[] {
    return this.inner.getStopTokens();
  }

  /**
   * Gets the service ID for the model
   */
  getServiceId(): number {
    return this.inner.getServiceId();
  }

  /**
   * Get the size of a KV page for this model
   */
  getKvPageSize(): number {
    return this.inner.getKvPageSize();
  }

  /**
   * Get the tokenizer for this model
   */
  getTokenizer(): Tokenizer {
    return new Tokenizer(this.inner);
  }

  /**
   * Returns the EOS (end-of-sequence) tokens as tokenized arrays
   */
  eosTokens(): Uint32Array[] {
    const tokenizer = new Tokenizer(this.inner);
    return this.getStopTokens().map(stopToken => tokenizer.tokenize(stopToken));
  }

  /**
   * Create a new command queue for this model
   */
  createQueue(): Queue {
    const queueResource = this.inner.createQueue();
    return new Queue(queueResource, this.getServiceId());
  }

  /**
   * Create a new context for this model
   * Note: This returns a Context that must be imported from './context.js'
   * to avoid circular dependencies
   */
  // createContext() is implemented via the Context constructor: new Context(model)
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
