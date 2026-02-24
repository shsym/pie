// Runtime wrapper functions for inferlet-core-runtime bindings
// These functions wrap the WASM component bindings with a clean TypeScript API

import * as runtime from 'inferlet:core/runtime';
import { awaitFuture } from './async-utils.js';

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
 * Parses an array of command line arguments into a JSON object.
 * 
 * Rules:
 * 1. Flags starting with '-' or '--' become keys.
 * 2. If a flag is followed by a value (not starting with '-'), it's a key-value pair.
 * 3. If a flag is followed by another flag or nothing, it is treated as boolean true.
 */
function parseArgs(args: string[]): Record<string, string | boolean> {
  const parsed: Record<string, string | boolean> = {};

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    // Only process flags
    if (arg.startsWith('-')) {
      // Remove leading dashes
      const key = arg.replace(/^-+/, '');

      // Peek at the next argument
      const nextArg = args[i + 1];

      // If next arg exists and is NOT a flag, it is the value
      if (nextArg && !nextArg.startsWith('-')) {
        parsed[key] = nextArg;
        i++; // Skip the next index as we just consumed it
      } else {
        // Otherwise, treat it as a boolean flag
        parsed[key] = true;
      }
    }
  }

  return parsed;
}

/**
 * Retrieves POSIX-style CLI arguments passed to the inferlet from the remote user client,
 * parsed into an object with key-value pairs.
 */
export function getArguments(): Record<string, string | boolean> {
  return parseArgs(runtime.getArguments());
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
  return awaitFuture(runtime.debugQuery(query), 'debugQuery result was undefined');
}
