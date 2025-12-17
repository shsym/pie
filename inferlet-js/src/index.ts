// inferlet-js - JavaScript library for writing Pie inferlets
// This mirrors the Rust inferlet crate API

export const VERSION = '0.1.0';

// Re-export runtime functions
export {
  getVersion,
  getInstanceId,
  getArguments,
  setReturn,
  debugQuery,
} from './runtime.js';

// Re-export Sampler
export { Sampler } from './sampler.js';
export type { SamplerType } from './sampler.js';

// Re-export StopCondition
export {
  MaxLen,
  EndsWith,
  AnyEndsWith,
  Or,
  maxLen,
  endsWith,
  endsWithAny,
} from './stop-condition.js';
export type { StopCondition } from './stop-condition.js';

// Re-export Tokenizer
export { Tokenizer } from './tokenizer.js';

// Re-export Model and Queue
export { Model, Queue, getModel, getAllModels, getAutoModel, getAllModelsWithTraits } from './model.js';

// Re-export ChatFormatter
export { ChatFormatter } from './chat.js';
export type { ToolCall } from './chat.js';

// Re-export Brle
export { Brle, causalMask } from './brle.js';

// Re-export ForwardPass and related types
export { ForwardPass, KvPage, Resource, causalMask as forwardCausalMask } from './forward.js';
export type { Distribution, ForwardPassResult, Forward } from './forward.js';

// Re-export Context
export { Context } from './context.js';

// Re-export messaging functions
export { send, receive, sendBlob, receiveBlob, broadcast, subscribe, Blob } from './messaging.js';

// Re-export KVS functions
export { storeGet, storeSet, storeDelete, storeExists, storeListKeys } from './kvs.js';
