// inferlet-js/test/mocks/inferlet-core-message.js
// Mock implementation of inferlet:core/message

import { Pollable } from './wasi-io-poll.js';
import { Blob, BlobResult } from './inferlet-core-common.js';

const sentMessages = [];
const messageQueue = [];
const blobQueue = [];
const subscriptions = new Map();

class ReceiveResult {
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

class ReceiveBlobResult {
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

class SubscribeResult {
  #topic;

  constructor(topic) {
    this.#topic = topic;
  }

  pollable() {
    return new Pollable(true);
  }

  get() {
    const messages = subscriptions.get(this.#topic) || [];
    return messages.shift() || `mock-subscription-${this.#topic}`;
  }
}

export function send(msg) {
  sentMessages.push(msg);
  // Also log for visibility during tests
  if (typeof process !== 'undefined' && process.stdout) {
    process.stdout.write(msg);
    if (!msg.endsWith('\n')) {
      process.stdout.write('\n');
    }
  }
}

export function receive() {
  const msg = messageQueue.shift() || 'mock-user-input';
  return new ReceiveResult(msg);
}

export function sendBlob(blob) {
  // No-op in mock
}

export function receiveBlob() {
  const blob = blobQueue.shift() || new Blob(new Uint8Array([0, 1, 2, 3]));
  return new ReceiveBlobResult(blob);
}

export function broadcast(topic, msg) {
  if (!subscriptions.has(topic)) {
    subscriptions.set(topic, []);
  }
  subscriptions.get(topic).push(msg);
}

export function subscribe(topic) {
  return new SubscribeResult(topic);
}

// Test helpers
export function __getSentMessages() {
  return [...sentMessages];
}

export function __queueMessage(msg) {
  messageQueue.push(msg);
}

export function __queueBlob(blob) {
  blobQueue.push(blob);
}

export function __resetMockState() {
  sentMessages.length = 0;
  messageQueue.length = 0;
  blobQueue.length = 0;
  subscriptions.clear();
}
