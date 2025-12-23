/** @module Interface inferlet:core/runtime **/
/**
 * Returns the runtime version string
 */
export function getVersion(): string;
/**
 * Returns a unique identifier for the running instance
 */
export function getInstanceId(): string;
/**
 * Retrieves POSIX-style CLI arguments passed to the inferlet from the remote user client
 */
export function getArguments(): Array<string>;
/**
 * Sets the return value for the inferlet
 */
export function setReturn(value: string): void;
/**
 * Retrieve a model by name; returns None if not found
 */
export function getModel(name: string): Model | undefined;
/**
 * Get a list of all available model names
 */
export function getAllModels(): Array<string>;
/**
 * Get names of models that have all specified traits (e.g. "input_text", "tokenize")
 */
export function getAllModelsWithTraits(traits: Array<string>): Array<string>;
/**
 * Executes a debug command and returns the result as a string
 */
export function debugQuery(query: string): DebugQueryResult;
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type Model = import('./inferlet-core-common.js').Model;
export type DebugQueryResult = import('./inferlet-core-common.js').DebugQueryResult;
