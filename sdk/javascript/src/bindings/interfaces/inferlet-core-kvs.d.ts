/** @module Interface inferlet:core/kvs **/
/**
 * Retrieves a value from the persistent store for a given key.
 * Returns none if the key does not exist.
 */
export function storeGet(key: string): string | undefined;
/**
 * Sets a value in the persistent store for a given key.
 * This will create a new entry or overwrite an existing one.
 */
export function storeSet(key: string, value: string): void;
/**
 * Deletes a key-value pair from the store.
 */
export function storeDelete(key: string): void;
/**
 * Checks if a key exists in the store.
 */
export function storeExists(key: string): boolean;
/**
 * Returns a list of all keys currently in the store.
 */
export function storeListKeys(): Array<string>;
