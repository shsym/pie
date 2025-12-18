// Runner for executing transpiled WASM components
import { pathToFileURL } from 'url';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { createHostMocks, type MockOptions, resetResourceIds } from './host-mocks.js';

export interface RunResult {
  /** Whether the run succeeded */
  success: boolean;
  /** Error message if failed */
  error?: string;
  /** Captured output from send() calls */
  outputs: string[];
  /** Execution duration in ms */
  duration: number;
}

/**
 * Run a transpiled WASM component with mock host functions.
 *
 * @param modulePath - Path to the transpiled JS module (with instantiate export)
 * @param mockOptions - Options for configuring the mock host functions
 * @returns Run result with outputs and status
 */
export async function runInferlet(
  modulePath: string,
  mockOptions: MockOptions = {}
): Promise<RunResult> {
  // Reset resource IDs for isolation
  resetResourceIds();

  // Create output capture array
  const outputCapture: string[] = [];
  const options = {
    ...mockOptions,
    outputCapture,
  };

  const startTime = Date.now();

  try {
    // Dynamic import of the transpiled module
    // Use file:// URL for Windows compatibility
    const moduleUrl = pathToFileURL(modulePath).href;
    const module = await import(moduleUrl);

    // jco generates an instantiate function when using --instantiation async
    // Signature: instantiate(getCoreModule, imports, instantiateCore?)
    if (typeof module.instantiate !== 'function') {
      throw new Error(
        `Module does not export instantiate function. ` +
          `Available exports: ${Object.keys(module).join(', ')}`
      );
    }

    // Create host mocks
    const hostImports = createHostMocks(options);

    // Get the directory containing the transpiled files
    const moduleDir = dirname(modulePath);

    // Create getCoreModule function to load .core.wasm files
    const getCoreModule = async (name: string) => {
      // Core modules are named like: test.core.wasm, test.core2.wasm, etc.
      const corePath = join(moduleDir, name);
      const wasmBytes = readFileSync(corePath);
      return WebAssembly.compile(wasmBytes);
    };

    // Build the imports object in the format jco expects
    const imports = buildImports(hostImports);

    // Instantiate the component with core module loader and imports
    const instance = await module.instantiate(getCoreModule, imports);

    // The exported 'run' interface contains the run() function
    // Structure: instance.run.run() or instance['inferlet:core/run'].run()
    let runFn: (() => Promise<{ tag: string; val?: string }>) | undefined;

    if (instance.run?.run) {
      runFn = instance.run.run;
    } else if (instance['inferlet:core/run']?.run) {
      runFn = instance['inferlet:core/run'].run;
    }

    if (!runFn) {
      throw new Error(
        `Instance does not export run function. ` +
          `Available exports: ${JSON.stringify(Object.keys(instance))}`
      );
    }

    // Execute the inferlet
    const result = await runFn();

    const duration = Date.now() - startTime;

    // Check result - WIT result<_, string> has tag 'ok' or 'err'
    if (result === undefined || result === null) {
      return {
        success: true,
        outputs: outputCapture,
        duration,
      };
    }

    if (result.tag === 'err') {
      return {
        success: false,
        error: result.val ?? 'Unknown error',
        outputs: outputCapture,
        duration,
      };
    }

    return {
      success: true,
      outputs: outputCapture,
      duration,
    };
  } catch (error: any) {
    const duration = Date.now() - startTime;
    return {
      success: false,
      error: error.stack ?? error.message ?? String(error),
      outputs: outputCapture,
      duration,
    };
  }
}

/**
 * Build imports object in the format jco expects.
 * jco expects imports as a nested object: { 'namespace:package/interface': { function: impl } }
 */
function buildImports(hostImports: Record<string, any>): Record<string, any> {
  return hostImports;
}

/**
 * Build, transpile, and run an inferlet in one step.
 * Convenience function for simple test cases.
 */
export async function buildAndRun(
  inputDir: string,
  mockOptions: MockOptions = {}
): Promise<RunResult & { buildDuration: number; transpileDuration: number }> {
  const { buildInferlet } = await import('./build.js');
  const { transpileComponent } = await import('./transpile.js');
  const { tmpdir } = await import('os');
  const { join } = await import('path');
  const { mkdtempSync } = await import('fs');

  // Create temp directory for outputs
  const tempDir = mkdtempSync(join(tmpdir(), 'inferlet-test-'));
  const wasmPath = join(tempDir, 'test.wasm');
  const transpileDir = join(tempDir, 'transpiled');

  // Build
  const buildResult = await buildInferlet(inputDir, wasmPath);

  // Transpile
  const transpileResult = await transpileComponent(wasmPath, transpileDir);

  // Run
  const runResult = await runInferlet(transpileResult.modulePath, mockOptions);

  return {
    ...runResult,
    buildDuration: buildResult.duration,
    transpileDuration: transpileResult.duration,
  };
}
