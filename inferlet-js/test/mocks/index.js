// inferlet-js/test/mocks/index.js
// Re-export all mocks for convenient import mapping

export * as poll from './wasi-io-poll.js';
export * as common from './inferlet-core-common.js';
export * as runtime from './inferlet-core-runtime.js';
export * as tokenize from './inferlet-core-tokenize.js';
export * as forward from './inferlet-core-forward.js';
export * as kvs from './inferlet-core-kvs.js';
export * as message from './inferlet-core-message.js';

// Combined reset function
export async function resetAllMocks() {
  const { __resetMockState: resetCommon } = await import('./inferlet-core-common.js');
  const { __resetMockState: resetRuntime } = await import('./inferlet-core-runtime.js');
  const { __resetMockState: resetForward } = await import('./inferlet-core-forward.js');
  const { __resetMockState: resetKvs } = await import('./inferlet-core-kvs.js');
  const { __resetMockState: resetMessage } = await import('./inferlet-core-message.js');

  resetCommon?.();
  resetRuntime?.();
  resetForward?.();
  resetKvs?.();
  resetMessage?.();
}
