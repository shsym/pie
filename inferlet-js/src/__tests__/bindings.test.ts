import { describe, it, expect } from 'vitest';

// These tests verify that the WIT binding modules export the expected structure.
// They use dynamic imports to catch runtime issues, not just compile-time checks.
// Note: WIT imports are resolved at runtime by vitest aliases to mock implementations

describe('WIT Bindings Runtime Verification', () => {
  describe('inferlet-core-runtime bindings', () => {
    it('should export getVersion function', async () => {
      // This uses the mocked version via vitest aliases
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const runtime = await import('inferlet:core/runtime');
      expect(typeof runtime.getVersion).toBe('function');
    });

    it('should export getArguments function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const runtime = await import('inferlet:core/runtime');
      expect(typeof runtime.getArguments).toBe('function');
    });

    it('should export getModel function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const runtime = await import('inferlet:core/runtime');
      expect(typeof runtime.getModel).toBe('function');
    });

    it('should export getAllModels function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const runtime = await import('inferlet:core/runtime');
      expect(typeof runtime.getAllModels).toBe('function');
    });
  });

  describe('inferlet-core-kvs bindings', () => {
    it('should export storeGet function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const kvs = await import('inferlet:core/kvs');
      expect(typeof kvs.storeGet).toBe('function');
    });

    it('should export storeSet function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const kvs = await import('inferlet:core/kvs');
      expect(typeof kvs.storeSet).toBe('function');
    });

    it('should export storeDelete function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const kvs = await import('inferlet:core/kvs');
      expect(typeof kvs.storeDelete).toBe('function');
    });

    it('should export storeExists function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const kvs = await import('inferlet:core/kvs');
      expect(typeof kvs.storeExists).toBe('function');
    });

    it('should export storeListKeys function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const kvs = await import('inferlet:core/kvs');
      expect(typeof kvs.storeListKeys).toBe('function');
    });
  });

  describe('inferlet-core-message bindings', () => {
    it('should export send function', async () => {
      // @ts-expect-error - WIT imports resolved by vitest at runtime
      const message = await import('inferlet:core/message');
      expect(typeof message.send).toBe('function');
    });
  });
});
