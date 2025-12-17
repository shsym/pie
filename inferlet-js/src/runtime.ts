// Runtime wrapper functions for inferlet-core-runtime bindings
// These functions wrap the WASM component bindings with a clean TypeScript API

import * as runtime from './bindings/interfaces/inferlet-core-runtime.js';
import type { Pollable } from './bindings/interfaces/wasi-io-poll.js';
import type { DebugQueryResult as DebugQueryResultResource } from './bindings/interfaces/inferlet-core-common.js';

/**
 * Returns the runtime version string
 */
export function getVersion(): string {
  return runtime.getVersion();
}

/**
 * Returns a unique identifier for the running instance
 */
export function getInstanceId(): string {
  return runtime.getInstanceId();
}

/**
 * Retrieves POSIX-style CLI arguments passed to the inferlet from the remote user client
 */
export function getArguments(): string[] {
  return runtime.getArguments();
}

/**
 * Sets the return value for the inferlet
 */
export function setReturn(value: string): void {
  runtime.setReturn(value);
}

/**
 * Executes a debug command and returns the result as a string (async)
 * This function handles the pollable resource to wait for the result
 */
export async function debugQuery(query: string): Promise<string> {
  const result: DebugQueryResultResource = runtime.debugQuery(query);

  // Get the pollable for async waiting
  const pollable: Pollable = result.pollable();

  // Block until the result is ready
  pollable.block();

  // Get the result (should be available now)
  const value = result.get();

  if (value === undefined) {
    throw new Error('debugQuery result was undefined after pollable was ready');
  }

  return value;
}
