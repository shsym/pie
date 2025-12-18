// inferlet-js/test/wasm/__tests__/context.wasm.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { __resetMockState as resetRuntime, __setMockModels } from '../../mocks/inferlet-core-runtime.js';
import { __resetMockState as resetCommon } from '../../mocks/inferlet-core-common.js';
import { __resetMockState as resetForward } from '../../mocks/inferlet-core-forward.js';

// Import the actual inferlet-js modules (will use mocked WIT imports)
import { getAutoModel } from '../../../src/model.js';
import { Context } from '../../../src/context.js';

describe('Context (WASM mock integration)', () => {
  beforeEach(() => {
    resetRuntime();
    resetCommon();
    resetForward();
    __setMockModels(['mock-llama-3.2-1b']);
  });

  describe('Context creation', () => {
    it('should create context from model', () => {
      const model = getAutoModel();
      expect(model).toBeDefined();
      expect(model.name).toBe('mock-llama-3.2-1b');

      const ctx = new Context(model);
      expect(ctx).toBeDefined();
      expect(ctx.queue).toBeDefined();
      expect(ctx.model).toBe(model);
    });

    it('should have tokenizer from model', () => {
      const model = getAutoModel();
      const ctx = new Context(model);
      expect(ctx.tokenizer).toBeDefined();
    });
  });

  describe('Context.fill()', () => {
    it('should fill context with text', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fill('Hello world');

      // Tokens should be in pending buffer
      expect(ctx.tokenIdsPending.length).toBeGreaterThan(0);
    });

    it('should fill context with tokens', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fillTokens([1, 2, 3, 4, 5]);

      expect(ctx.tokenIdsPending).toEqual([1, 2, 3, 4, 5]);
    });

    it('should fill context with single token', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fillToken(42);

      expect(ctx.tokenIdsPending).toEqual([42]);
    });
  });

  describe('Context.fork()', () => {
    it('should fork context', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fill('Hello world');

      // Fork the context
      const forked = ctx.fork();

      expect(forked).toBeDefined();
      expect(forked).not.toBe(ctx);
    });

    it('should create independent branches after fork', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fill('Base content');

      const fork1 = ctx.fork();
      const fork2 = ctx.fork();

      fork1.fill(' branch 1');
      fork2.fill(' branch 2');

      // Each fork should have independent pending tokens
      expect(fork1.tokenIdsPending).not.toEqual(fork2.tokenIdsPending);
    });
  });

  describe('Context.release()', () => {
    it('should release context resources', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fill('Test content');

      // Should not throw
      expect(() => ctx.release()).not.toThrow();
    });

    it('should be safe to call multiple times', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.release();
      expect(() => ctx.release()).not.toThrow();
    });
  });

  describe('Chat formatter integration', () => {
    it('should fill system message', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fillSystem('You are a helpful assistant.');

      expect(ctx.tokenIdsPending.length).toBeGreaterThan(0);
    });

    it('should fill user message', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fillUser('Hello!');

      expect(ctx.tokenIdsPending.length).toBeGreaterThan(0);
    });

    it('should fill assistant message', () => {
      const model = getAutoModel();
      const ctx = new Context(model);

      ctx.fillAssistant('Hi there!');

      expect(ctx.tokenIdsPending.length).toBeGreaterThan(0);
    });
  });
});
