// inferlet-js/test/mocks/wasi-io-poll.js
// Mock implementation of wasi:io/poll@0.2.4

export class Pollable {
  #ready = true;

  constructor(ready = true) {
    this.#ready = ready;
  }

  ready() {
    return this.#ready;
  }

  block() {
    // In mock, always immediately ready - no actual blocking
    this.#ready = true;
  }
}

export function poll(pollables) {
  // Return indices of all ready pollables
  const ready = [];
  for (let i = 0; i < pollables.length; i++) {
    if (pollables[i].ready()) {
      ready.push(i);
    }
  }
  return new Uint32Array(ready.length > 0 ? ready : [0]);
}
