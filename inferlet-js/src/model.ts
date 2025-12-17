// Model and Queue wrapper classes for inferlet-core bindings
// These classes wrap the WASM component bindings with a clean TypeScript API

import * as runtime from './bindings/interfaces/inferlet-core-runtime.js';
import type {
  Model as ModelResource,
  Queue as QueueResource,
  Priority,
  SynchronizationResult,
  DebugQueryResult,
} from './bindings/interfaces/inferlet-core-common.js';
import type { Pollable } from './bindings/interfaces/wasi-io-poll.js';
import { Tokenizer } from './tokenizer.js';

/**
 * Represents a command queue for a specific model instance.
 * Queues manage the execution of commands and resource allocation.
 */
export class Queue {
  private readonly inner: QueueResource;
  private readonly serviceId: number;

  constructor(inner: QueueResource, serviceId: number) {
    this.inner = inner;
    this.serviceId = serviceId;
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

  // Note: createContext() will be added in Task 10
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
