// inferlet-js/test/mocks/inferlet-core-kvs.js
// Mock implementation of inferlet:core/kvs

const store = new Map();

export function storeGet(key) {
  return store.get(key);
}

export function storeSet(key, value) {
  store.set(key, value);
}

export function storeDelete(key) {
  store.delete(key);
}

export function storeExists(key) {
  return store.has(key);
}

export function storeListKeys() {
  return [...store.keys()];
}

// Test helper
export function __resetMockState() {
  store.clear();
}

export function __getStore() {
  return new Map(store);
}
