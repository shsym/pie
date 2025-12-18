// Transpile harness for converting WASM components to executable JS
import { execSync } from 'child_process';
import { existsSync, mkdirSync, readdirSync } from 'fs';
import { resolve, join } from 'path';

export interface TranspileOptions {
  /** Generate TypeScript definitions */
  typescript?: boolean;
  /** Timeout in milliseconds */
  timeout?: number;
}

export interface TranspileResult {
  /** Path to the main module (instantiate function) */
  modulePath: string;
  /** Directory containing all generated files */
  outDir: string;
  /** List of generated files */
  files: string[];
  /** Transpilation duration in ms */
  duration: number;
}

/**
 * Transpile a WASM component to JS using jco.
 *
 * Uses --instantiation async mode to generate an instantiate() function
 * that accepts custom import implementations.
 *
 * @param wasmPath - Path to the .wasm component file
 * @param outDir - Directory where transpiled files will be written
 * @param options - Transpile options
 * @returns Transpile result with paths and metadata
 * @throws Error if transpilation fails
 */
export async function transpileComponent(
  wasmPath: string,
  outDir: string,
  options: TranspileOptions = {}
): Promise<TranspileResult> {
  const { typescript = false, timeout = 60000 } = options;

  // Verify input exists
  if (!existsSync(wasmPath)) {
    throw new Error(`WASM file not found: ${wasmPath}`);
  }

  // Ensure output directory exists
  if (!existsSync(outDir)) {
    mkdirSync(outDir, { recursive: true });
  }

  // Build jco transpile command
  // --instantiation async: Generate instantiate() that accepts imports
  // --no-wasi-shim: Don't auto-rewrite WASI imports (we provide our own)
  const args = [
    'jco',
    'transpile',
    wasmPath,
    '-o',
    outDir,
    '--instantiation',
    'async',
    '--no-wasi-shim',
  ];

  if (!typescript) {
    args.push('--no-typescript');
  }

  const startTime = Date.now();

  try {
    execSync(`npx ${args.join(' ')}`, {
      timeout,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: resolve(__dirname, '../../..'), // inferlet-js root for npx
    });
  } catch (error: any) {
    const stderr = error.stderr?.toString() || '';
    const stdout = error.stdout?.toString() || '';
    throw new Error(
      `jco transpile failed:\n` +
        `  Input: ${wasmPath}\n` +
        `  Output: ${outDir}\n` +
        `  Exit code: ${error.status}\n` +
        `  Stderr: ${stderr}\n` +
        `  Stdout: ${stdout}`
    );
  }

  const duration = Date.now() - startTime;

  // Find generated files
  const files = readdirSync(outDir);
  if (files.length === 0) {
    throw new Error(`Transpile succeeded but no files generated in: ${outDir}`);
  }

  // Find the main module - jco generates component.js by default
  // The instantiate function is exported from this module
  const mainCandidates = files.filter(
    (f) => f.endsWith('.js') && !f.endsWith('.core.wasm')
  );

  if (mainCandidates.length === 0) {
    throw new Error(
      `No JS module found in transpile output: ${outDir}\nFiles: ${files.join(', ')}`
    );
  }

  // Prefer the shortest name (usually component.js or similar)
  mainCandidates.sort((a, b) => a.length - b.length);
  const modulePath = join(outDir, mainCandidates[0]);

  return {
    modulePath,
    outDir,
    files,
    duration,
  };
}

/**
 * Check if jco is available.
 */
export function isJcoAvailable(): boolean {
  try {
    execSync('npx jco --version', {
      stdio: 'pipe',
      cwd: resolve(__dirname, '../../..'),
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Get jco version.
 */
export function getJcoVersion(): string | null {
  try {
    const output = execSync('npx jco --version', {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: resolve(__dirname, '../../..'),
    });
    return output.trim();
  } catch {
    return null;
  }
}
