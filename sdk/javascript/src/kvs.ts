// Key-Value Store functions for persistent storage.
// Mirrors the Rust KVS functions from inferlet/src/lib.rs

import * as kvs from 'inferlet:core/kvs';

/**
 * Retrieves a value from the persistent store for a given key.
 * @param key The key to look up
 * @returns The value if found, or undefined if the key does not exist
 */
export function storeGet(key: string): string | undefined {
  return kvs.storeGet(key);
}

/**
 * Sets a value in the persistent store for a given key.
 * This will create a new entry or overwrite an existing one.
 * @param key The key to set
 * @param value The value to store
 */
export function storeSet(key: string, value: string): void {
  kvs.storeSet(key, value);
}

/**
 * Deletes a key-value pair from the store.
 * If the key does not exist, this function does nothing.
 * @param key The key to delete
 */
export function storeDelete(key: string): void {
  kvs.storeDelete(key);
}

/**
 * Checks if a key exists in the store.
 * @param key The key to check
 * @returns true if the key exists, false otherwise
 */
export function storeExists(key: string): boolean {
  return kvs.storeExists(key);
}

/**
 * Returns a list of all keys currently in the store.
 * @returns Array of all keys
 */
export function storeListKeys(): string[] {
  return [...kvs.storeListKeys()];
}
