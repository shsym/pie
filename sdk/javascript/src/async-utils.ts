// Async utilities for WASI pollable futures

/**
 * Minimal pollable interface (matches wasi:io/poll Pollable)
 */
interface Pollable {
  ready(): boolean;
  block(): void;
}

/**
 * Generic future interface for WASI async operations.
 * Matches WIT resources like receive-result, subscription, blob-result.
 */
export interface WasiFuture<T> {
  pollable(): Pollable;
  get(): T | undefined;
}

/**
 * Awaits a WASI future by blocking on its pollable.
 * @param future The WASI future to await
 * @param errorMessage Error message if result is undefined
 * @returns The resolved value
 * @throws Error if the future returns undefined
 */
export function awaitFuture<T>(future: WasiFuture<T>, errorMessage: string): T {
  const pollable = future.pollable();
  pollable.block();

  const result = future.get();
  if (result === undefined) {
    throw new Error(errorMessage);
  }
  return result;
}
