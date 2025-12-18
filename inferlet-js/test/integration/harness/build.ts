// Build harness for compiling inferlets to WASM
import { execSync } from 'child_process';
import { existsSync, mkdirSync, statSync } from 'fs';
import { dirname, resolve } from 'path';

export interface BuildOptions {
  debug?: boolean;
  timeout?: number;
}

export interface BuildResult {
  wasmPath: string;
  size: number;
  duration: number;
}

/**
 * Build an inferlet to WASM using pie-cli.
 *
 * @param inputDir - Directory containing the inferlet (with package.json) or path to entry file
 * @param outputWasm - Path where the .wasm file should be written
 * @param options - Build options
 * @returns Build result with path and metadata
 * @throws Error if build fails
 */
export async function buildInferlet(
  inputDir: string,
  outputWasm: string,
  options: BuildOptions = {}
): Promise<BuildResult> {
  const { debug = false, timeout = 120000 } = options;

  // Ensure output directory exists
  const outDir = dirname(outputWasm);
  if (!existsSync(outDir)) {
    mkdirSync(outDir, { recursive: true });
  }

  // Build command
  const args = ['build', inputDir, '-o', outputWasm];
  if (debug) {
    args.push('--debug');
  }

  const startTime = Date.now();

  try {
    execSync(`pie-cli ${args.join(' ')}`, {
      timeout,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      // Set PIE_HOME to help pie-cli find inferlet-js
      env: {
        ...process.env,
        PIE_HOME: resolve(__dirname, '../../..'),
      },
    });
  } catch (error: any) {
    const stderr = error.stderr?.toString() || '';
    const stdout = error.stdout?.toString() || '';
    throw new Error(
      `pie-cli build failed:\n` +
        `  Input: ${inputDir}\n` +
        `  Output: ${outputWasm}\n` +
        `  Exit code: ${error.status}\n` +
        `  Stderr: ${stderr}\n` +
        `  Stdout: ${stdout}`
    );
  }

  const duration = Date.now() - startTime;

  // Verify output exists
  if (!existsSync(outputWasm)) {
    throw new Error(`Build succeeded but output file not found: ${outputWasm}`);
  }

  const stats = statSync(outputWasm);
  if (stats.size === 0) {
    throw new Error(`Build produced empty WASM file: ${outputWasm}`);
  }

  return {
    wasmPath: outputWasm,
    size: stats.size,
    duration,
  };
}

/**
 * Check if pie-cli is available in PATH.
 */
export function isPieCliAvailable(): boolean {
  try {
    execSync('which pie-cli', { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Get pie-cli version.
 */
export function getPieCliVersion(): string | null {
  try {
    const output = execSync('pie-cli --version', {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return output.trim();
  } catch {
    return null;
  }
}
