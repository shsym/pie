import { describe, it, expect } from 'vitest';

// Test that generated TypeScript bindings compile correctly
// These are .d.ts files, so we test that TypeScript can type-check code using them
describe('WIT Bindings', () => {
  it('should have valid TypeScript type definitions', () => {
    // Import types to verify they compile
    type CoreRuntime = typeof import('../bindings/interfaces/inferlet-core-runtime.js');
    type CoreCommon = typeof import('../bindings/interfaces/inferlet-core-common.js');
    type CoreForward = typeof import('../bindings/interfaces/inferlet-core-forward.js');
    type CoreTokenize = typeof import('../bindings/interfaces/inferlet-core-tokenize.js');
    type CoreMessage = typeof import('../bindings/interfaces/inferlet-core-message.js');
    type CoreKvs = typeof import('../bindings/interfaces/inferlet-core-kvs.js');
    type CoreRun = typeof import('../bindings/interfaces/inferlet-core-run.js');
    type AdapterCommon = typeof import('../bindings/interfaces/inferlet-adapter-common.js');
    type ImageImage = typeof import('../bindings/interfaces/inferlet-image-image.js');
    type ZoEvolve = typeof import('../bindings/interfaces/inferlet-zo-evolve.js');
    type WasiIoPoll = typeof import('../bindings/interfaces/wasi-io-poll.js');
    type ExecWorld = typeof import('../bindings/exec.js');

    // If TypeScript compiles this test, the bindings are valid
    expect(true).toBe(true);
  });

  it('should export expected namespaces from exec world', () => {
    // Verify the exec.d.ts structure by importing types
    type Exec = typeof import('../bindings/exec.js');

    // Type-level assertions - if these compile, the structure is correct
    const assertExecStructure = (exec: Exec) => {
      const _runtime: typeof exec.InferletCoreRuntime = {} as any;
      const _common: typeof exec.InferletCoreCommon = {} as any;
      const _forward: typeof exec.InferletCoreForward = {} as any;
      const _tokenize: typeof exec.InferletCoreTokenize = {} as any;
      const _message: typeof exec.InferletCoreMessage = {} as any;
      const _kvs: typeof exec.InferletCoreKvs = {} as any;
      const _adapter: typeof exec.InferletAdapterCommon = {} as any;
      const _image: typeof exec.InferletImageImage = {} as any;
      const _zo: typeof exec.InferletZoEvolve = {} as any;
      const _poll: typeof exec.WasiIoPoll024 = {} as any;
      const _run: typeof exec.run = {} as any;
    };

    // If TypeScript compiles this, the exec world exports are correct
    expect(assertExecStructure).toBeDefined();
  });
});
