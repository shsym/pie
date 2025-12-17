/** @module Interface inferlet:core/common **/
/**
 * resources
 */
export function allocateResources(queue: Queue, resourceType: number, count: number): Uint32Array;
export function deallocateResources(queue: Queue, resourceType: number, ptrs: Uint32Array): void;
export function getAllExportedResources(queue: Queue, resourceType: number): Array<[string, number]>;
export function releaseExportedResources(queue: Queue, resourceType: number, name: string): void;
export function exportResources(queue: Queue, resourceType: number, ptrs: Uint32Array, name: string): void;
export function importResources(queue: Queue, resourceType: number, name: string): Uint32Array;
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Pointer = number;
/**
 * Defines task priority levels
 * # Variants
 * 
 * ## `"low"`
 * 
 * ## `"normal"`
 * 
 * Lowest priority
 * ## `"high"`
 * 
 * Default priority
 */
export type Priority = 'low' | 'normal' | 'high';

export class Blob {
  constructor(init: Uint8Array)
  read(offset: bigint, n: bigint): Uint8Array;
  size(): bigint;
}

export class BlobResult {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Pollable to check readiness
  */
  pollable(): Pollable;
  /**
  * Retrieves the message if available; None if not ready
  */
  get(): Blob | undefined;
}

export class DebugQueryResult {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  pollable(): Pollable;
  get(): string | undefined;
}

export class Model {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Returns the model's name (e.g. "llama-3.1-8b-instruct")
  */
  getName(): string;
  /**
  * Returns the full set of model traits
  */
  getTraits(): Array<string>;
  /**
  * Human-readable description of the model
  */
  getDescription(): string;
  /**
  * Returns the prompt formatting template in Tera
  */
  getPromptTemplate(): string;
  getStopTokens(): Array<string>;
  getServiceId(): number;
  /**
  * Get the size of a KV page
  */
  getKvPageSize(): number;
  createQueue(): Queue;
}

export class Queue {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  getServiceId(): number;
  /**
  * Begin synchronization process
  */
  synchronize(): SynchronizationResult;
  /**
  * Change the queue's priority
  */
  setPriority(priority: Priority): void;
  debugQuery(query: string): DebugQueryResult;
}

export class SynchronizationResult {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Returns a pollable for async readiness checks
  */
  pollable(): Pollable;
  get(): boolean | undefined;
}
