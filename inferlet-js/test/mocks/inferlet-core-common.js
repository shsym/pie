// inferlet-js/test/mocks/inferlet-core-common.js
// Mock implementation of inferlet:core/common

import { Pollable } from './wasi-io-poll.js';

let resourceCounter = 0;
const exportedResources = new Map();

export class Blob {
  #data;

  constructor(init) {
    this.#data = new Uint8Array(init);
  }

  static fromWit(init) {
    return new Blob(init);
  }

  read(offset, n) {
    const start = Number(offset);
    const len = Number(n);
    return this.#data.slice(start, start + len);
  }

  size() {
    return BigInt(this.#data.length);
  }
}

export class BlobResult {
  #value;

  constructor(value) {
    this.#value = value;
  }

  pollable() {
    return new Pollable(true);
  }

  get() {
    return this.#value;
  }
}

export class DebugQueryResult {
  #value;

  constructor(value) {
    this.#value = value;
  }

  pollable() {
    return new Pollable(true);
  }

  get() {
    return this.#value;
  }
}

export class SynchronizationResult {
  pollable() {
    return new Pollable(true);
  }

  get() {
    return true;
  }
}

export class Queue {
  #serviceId;

  constructor(serviceId = 1) {
    this.#serviceId = serviceId;
  }

  getServiceId() {
    return this.#serviceId;
  }

  synchronize() {
    return new SynchronizationResult();
  }

  setPriority(priority) {
    // No-op in mock
  }

  debugQuery(query) {
    return new DebugQueryResult(`mock-debug: ${query}`);
  }
}

export class Model {
  #name;
  #kvPageSize;

  constructor(name, kvPageSize = 256) {
    this.#name = name;
    this.#kvPageSize = kvPageSize;
  }

  getName() {
    return this.#name;
  }

  getTraits() {
    return ['input_text', 'output_text', 'tokenize'];
  }

  getDescription() {
    return `Mock model: ${this.#name}`;
  }

  getPromptTemplate() {
    return '{% for message in messages %}{% if message.role == "system" %}<|start_header_id|>system<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% elif message.role == "user" %}<|start_header_id|>user<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% elif message.role == "assistant" %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}';
  }

  getStopTokens() {
    return ['<|eot_id|>', '<|end_of_text|>'];
  }

  getServiceId() {
    return 1;
  }

  getKvPageSize() {
    return this.#kvPageSize;
  }

  createQueue() {
    return new Queue(this.getServiceId());
  }
}

// Resource management functions
export function allocateResources(queue, resourceType, count) {
  const ptrs = [];
  for (let i = 0; i < count; i++) {
    ptrs.push(++resourceCounter);
  }
  return new Uint32Array(ptrs);
}

export function deallocateResources(queue, resourceType, ptrs) {
  // No-op in mock - resources are virtual
}

export function getAllExportedResources(queue, resourceType) {
  const result = [];
  for (const [name, resources] of exportedResources) {
    if (resources.type === resourceType) {
      result.push([name, resources.ptrs.length]);
    }
  }
  return result;
}

export function releaseExportedResources(queue, resourceType, name) {
  exportedResources.delete(name);
}

export function exportResources(queue, resourceType, ptrs, name) {
  exportedResources.set(name, { type: resourceType, ptrs: Array.from(ptrs) });
}

export function importResources(queue, resourceType, name) {
  const resources = exportedResources.get(name);
  if (!resources || resources.type !== resourceType) {
    return new Uint32Array([]);
  }
  return new Uint32Array(resources.ptrs);
}

// Test helper to reset state
export function __resetMockState() {
  resourceCounter = 0;
  exportedResources.clear();
}
