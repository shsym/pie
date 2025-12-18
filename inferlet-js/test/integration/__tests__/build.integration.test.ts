// Integration tests for WASM build and execution
import { describe, it, expect, beforeAll } from 'vitest';
import { resolve, join } from 'path';
import { mkdtempSync, rmSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import {
  buildInferlet,
  isPieCliAvailable,
  getPieCliVersion,
} from '../harness/build.js';
import { transpileComponent, isJcoAvailable } from '../harness/transpile.js';
import { runInferlet } from '../harness/runner.js';

const FIXTURES_DIR = resolve(__dirname, '../fixtures');

// Skip all tests if required tools are not available
const pieCliAvailable = isPieCliAvailable();
const jcoAvailable = isJcoAvailable();

describe.skipIf(!pieCliAvailable || !jcoAvailable)(
  'WASM Integration Tests',
  () => {
    let tempDir: string;

    beforeAll(() => {
      // Create a temp directory for all test outputs
      tempDir = mkdtempSync(join(tmpdir(), 'inferlet-integration-'));

      console.log('Test environment:');
      console.log(`  pie-cli: ${getPieCliVersion() ?? 'not found'}`);
      console.log(`  Temp dir: ${tempDir}`);
    });

    describe('Build Infrastructure', () => {
      it('should have pie-cli available', () => {
        expect(isPieCliAvailable()).toBe(true);
      });

      it('should have jco available', () => {
        expect(isJcoAvailable()).toBe(true);
      });

      it('should build basic-echo inferlet to WASM', async () => {
        const inputDir = join(FIXTURES_DIR, 'basic-echo');
        const outputWasm = join(tempDir, 'basic-echo.wasm');

        const result = await buildInferlet(inputDir, outputWasm);

        expect(result.wasmPath).toBe(outputWasm);
        expect(result.size).toBeGreaterThan(0);
        expect(existsSync(outputWasm)).toBe(true);

        console.log(`  Build size: ${(result.size / 1024).toFixed(1)} KB`);
        console.log(`  Build time: ${result.duration} ms`);
      }, 60000); // 60s timeout for build

      it('should transpile WASM to JS', async () => {
        const wasmPath = join(tempDir, 'basic-echo.wasm');
        const transpileDir = join(tempDir, 'basic-echo-transpiled');

        // Skip if WASM wasn't built
        if (!existsSync(wasmPath)) {
          console.log('  Skipping: WASM not built');
          return;
        }

        const result = await transpileComponent(wasmPath, transpileDir);

        expect(result.modulePath).toBeTruthy();
        expect(result.files.length).toBeGreaterThan(0);
        expect(existsSync(result.modulePath)).toBe(true);

        console.log(`  Generated files: ${result.files.join(', ')}`);
        console.log(`  Transpile time: ${result.duration} ms`);
      }, 30000);
    });

    describe('basic-echo Execution', () => {
      let modulePath: string;

      beforeAll(async () => {
        // Build and transpile once for all tests in this suite
        const inputDir = join(FIXTURES_DIR, 'basic-echo');
        const wasmPath = join(tempDir, 'basic-echo-exec.wasm');
        const transpileDir = join(tempDir, 'basic-echo-exec-transpiled');

        await buildInferlet(inputDir, wasmPath);
        const transpileResult = await transpileComponent(wasmPath, transpileDir);
        modulePath = transpileResult.modulePath;
      }, 90000);

      it('should execute and capture output', async () => {
        const result = await runInferlet(modulePath, {
          args: ['hello', 'world'],
        });

        expect(result.success).toBe(true);
        expect(result.outputs).toContain('hello world');
      });

      it('should handle empty arguments', async () => {
        const result = await runInferlet(modulePath, {
          args: [],
        });

        expect(result.success).toBe(true);
        expect(result.outputs).toContain('no arguments');
      });

      it('should handle special characters in arguments', async () => {
        const result = await runInferlet(modulePath, {
          args: ['hello!', '@world', '#test'],
        });

        expect(result.success).toBe(true);
        expect(result.outputs.join('')).toContain('hello! @world #test');
      });
    });

    // TODO: Tokenizer integration tests are disabled because they require:
    // 1. WIT resource handle management - Model resource from getAutoModel() needs
    //    proper registration in jco's resource table
    // 2. Mock implementation of the full inference pipeline (forward pass, sampling)
    // 3. Significant infrastructure work to create proper host function bindings
    //
    // Current workaround: tokenizer behavior is tested via:
    // - Unit tests in src/__tests__/ using direct module imports
    // - WASM mock tests in test/wasm/__tests__/tokenizer.wasm.test.ts
    //
    // To re-enable: implement createResource/dropResource in host-mocks.ts
    // and register the Model handle before returning from getAutoModel mock.
    describe.skip('tokenizer-test Execution', () => {
      let modulePath: string;

      beforeAll(async () => {
        const inputDir = join(FIXTURES_DIR, 'tokenizer-test');
        const wasmPath = join(tempDir, 'tokenizer-test.wasm');
        const transpileDir = join(tempDir, 'tokenizer-test-transpiled');

        await buildInferlet(inputDir, wasmPath);
        const transpileResult = await transpileComponent(wasmPath, transpileDir);
        modulePath = transpileResult.modulePath;
      }, 90000);

      it('should tokenize and detokenize correctly', async () => {
        const result = await runInferlet(modulePath, {
          args: ['Hello'],
          models: {
            'mock-model': {
              name: 'mock-model',
              traits: ['input_text', 'output_text', 'tokenize'],
            },
          },
        });

        expect(result.success).toBe(true);
        expect(result.outputs.length).toBeGreaterThan(0);

        const output = JSON.parse(result.outputs.join(''));
        expect(output.original).toBe('Hello');
        expect(output.tokenCount).toBe(5); // H, e, l, l, o
        expect(output.decoded).toBe('Hello');
        expect(output.match).toBe(true);
      });

      it('should handle special tokens', async () => {
        const result = await runInferlet(modulePath, {
          args: ['<|eot_id|>'],
          models: {
            'mock-model': {
              name: 'mock-model',
              traits: ['input_text', 'output_text', 'tokenize'],
            },
          },
        });

        expect(result.success).toBe(true);

        const output = JSON.parse(result.outputs.join(''));
        expect(output.original).toBe('<|eot_id|>');
        expect(output.tokenCount).toBe(1); // Single special token
        expect(output.match).toBe(true);
      });
    });
  }
);

// Separate describe for when tools are not available
describe.skipIf(pieCliAvailable && jcoAvailable)(
  'WASM Integration Tests (skipped)',
  () => {
    it('should skip tests when tools are not available', () => {
      console.log('Skipping integration tests:');
      if (!pieCliAvailable) console.log('  - pie-cli not found in PATH');
      if (!jcoAvailable) console.log('  - jco not available');
      expect(true).toBe(true);
    });
  }
);
