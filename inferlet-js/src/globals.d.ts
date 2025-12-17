// Ambient declarations for auto-injected inferlet globals
// Users can reference this file in their tsconfig.json for type support
//
// Usage Option 1: Add to tsconfig.json
//   { "compilerOptions": { "types": ["inferlet/globals"] } }
//
// Usage Option 2: Triple-slash directive at top of file
//   /// <reference types="inferlet/globals" />

import type { SamplerType } from './sampler.js';
import type { StopCondition } from './stop-condition.js';
import type { ToolCall } from './chat.js';
import type { Distribution, ForwardPassResult, Forward } from './forward.js';

declare global {
  // Version constant
  const VERSION: string;

  // Runtime functions
  const getVersion: typeof import('./index.js').getVersion;
  const getInstanceId: typeof import('./index.js').getInstanceId;
  const getArguments: typeof import('./index.js').getArguments;
  const setReturn: typeof import('./index.js').setReturn;
  const debugQuery: typeof import('./index.js').debugQuery;

  // Model functions
  const getModel: typeof import('./index.js').getModel;
  const getAllModels: typeof import('./index.js').getAllModels;
  const getAutoModel: typeof import('./index.js').getAutoModel;
  const getAllModelsWithTraits: typeof import('./index.js').getAllModelsWithTraits;

  // Classes
  const Model: typeof import('./index.js').Model;
  const Queue: typeof import('./index.js').Queue;
  const Context: typeof import('./index.js').Context;
  const Sampler: typeof import('./index.js').Sampler;
  const Tokenizer: typeof import('./index.js').Tokenizer;
  const ChatFormatter: typeof import('./index.js').ChatFormatter;
  const Brle: typeof import('./index.js').Brle;
  const ForwardPass: typeof import('./index.js').ForwardPass;
  const KvPage: typeof import('./index.js').KvPage;
  const Resource: typeof import('./index.js').Resource;
  /**
   * Inferlet's Blob type - intentionally shadows the Web API Blob.
   * This Blob is used for messaging and runtime operations within the Inferlet environment.
   * WARNING: The native Web API Blob is not available in this context. Use the Inferlet
   * Blob API (sendBlob, receiveBlob) or convert data to compatible formats accordingly.
   */
  const Blob: typeof import('./index.js').Blob;

  // Stop condition functions
  const maxLen: typeof import('./index.js').maxLen;
  const endsWith: typeof import('./index.js').endsWith;
  const endsWithAny: typeof import('./index.js').endsWithAny;

  // Stop condition classes
  const MaxLen: typeof import('./index.js').MaxLen;
  const EndsWith: typeof import('./index.js').EndsWith;
  const AnyEndsWith: typeof import('./index.js').AnyEndsWith;
  const Or: typeof import('./index.js').Or;

  // Messaging functions
  const send: typeof import('./index.js').send;
  const receive: typeof import('./index.js').receive;
  const sendBlob: typeof import('./index.js').sendBlob;
  const receiveBlob: typeof import('./index.js').receiveBlob;
  const broadcast: typeof import('./index.js').broadcast;
  const subscribe: typeof import('./index.js').subscribe;

  // KVS functions
  const storeGet: typeof import('./index.js').storeGet;
  const storeSet: typeof import('./index.js').storeSet;
  const storeDelete: typeof import('./index.js').storeDelete;
  const storeExists: typeof import('./index.js').storeExists;
  const storeListKeys: typeof import('./index.js').storeListKeys;

  // Utilities
  const causalMask: typeof import('./index.js').causalMask;
  const forwardCausalMask: typeof import('./index.js').forwardCausalMask;
}

export {};
