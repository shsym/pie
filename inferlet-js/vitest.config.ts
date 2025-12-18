// inferlet-js/vitest.config.ts
import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    include: [
      'src/__tests__/**/*.test.ts',
      'test/wasm/__tests__/**/*.test.ts',
      'test/integration/__tests__/**/*.test.ts',
    ],
    environment: 'node',
    // Longer timeout for integration tests (build + transpile + run)
    testTimeout: 120000,
    alias: {
      // Map WIT imports to mock files for unit tests and mock-based WASM tests
      // Note: Integration tests don't use these - they run actual WASM
      'inferlet:core/runtime': resolve(__dirname, 'test/mocks/inferlet-core-runtime.js'),
      'inferlet:core/common': resolve(__dirname, 'test/mocks/inferlet-core-common.js'),
      'inferlet:core/forward': resolve(__dirname, 'test/mocks/inferlet-core-forward.js'),
      'inferlet:core/tokenize': resolve(__dirname, 'test/mocks/inferlet-core-tokenize.js'),
      'inferlet:core/kvs': resolve(__dirname, 'test/mocks/inferlet-core-kvs.js'),
      'inferlet:core/message': resolve(__dirname, 'test/mocks/inferlet-core-message.js'),
      'wasi:io/poll': resolve(__dirname, 'test/mocks/wasi-io-poll.js'),
    },
  },
});
