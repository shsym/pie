// Argument parsing library for inferlet-js
// Provides type-safe, schema-based CLI argument parsing with auto-generated help

import { getArguments } from './runtime.js';
import { send } from './messaging.js';

/**
 * Base option definition
 */
interface BaseOption {
  /** Help text description */
  description: string;
  /** Single-character short form (e.g., 'v' for -v) */
  alias?: string;
}

/**
 * String option definition
 */
export interface StringOption extends BaseOption {
  type: 'string';
  /** Default value (makes option optional) */
  default?: string;
  /** Allowed values */
  choices?: readonly string[];
}

/**
 * Number option definition
 */
export interface NumberOption extends BaseOption {
  type: 'number';
  /** Default value (makes option optional) */
  default?: number;
  /** Minimum value */
  min?: number;
  /** Maximum value */
  max?: number;
  /** Allowed values */
  choices?: readonly number[];
}

/**
 * Boolean option definition
 */
export interface BooleanOption extends BaseOption {
  type: 'boolean';
  /** Default value (defaults to false if not specified) */
  default?: boolean;
}

/**
 * Any option type
 */
export type ArgOption = StringOption | NumberOption | BooleanOption;

/**
 * Schema for argument definitions
 */
export type ArgSchema = {
  readonly [key: string]: ArgOption;
};

/**
 * Infer the value type from an option definition
 */
type InferOptionValue<T extends ArgOption> = T extends StringOption
  ? T['choices'] extends readonly string[]
    ? T['choices'][number]
    : string
  : T extends NumberOption
    ? T['choices'] extends readonly number[]
      ? T['choices'][number]
      : number
    : T extends BooleanOption
      ? boolean
      : never;

/**
 * Infer the parsed args type from a schema.
 * Includes help and helpText for checking if help was requested.
 */
export type ParsedArgs<T extends ArgSchema> = {
  [K in keyof T]: InferOptionValue<T[K]>;
} & {
  /** True if --help or -h was passed */
  help: boolean;
  /** Generated help text */
  helpText: string;
};

/**
 * Convert camelCase to kebab-case
 */
function toKebabCase(str: string): string {
  return str.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
}

/**
 * Generate help text from schema
 */
function generateHelpText<T extends ArgSchema>(schema: T): string {
  const lines: string[] = ['Usage: [inferlet] [OPTIONS]', '', 'Options:'];

  // Calculate max option length for alignment
  const optionStrings: Array<{ option: string; description: string }> = [];

  for (const [key, opt] of Object.entries(schema)) {
    const kebabName = toKebabCase(key);
    let optionStr = '';

    if (opt.alias) {
      optionStr = `-${opt.alias}, --${kebabName}`;
    } else {
      optionStr = `    --${kebabName}`;
    }

    if (opt.type !== 'boolean') {
      optionStr += ` <${opt.type}>`;
    }

    let desc = opt.description;

    // Add choices info
    if ('choices' in opt && opt.choices) {
      desc += ` [${opt.choices.join('|')}]`;
    }

    // Add range info for numbers
    if (opt.type === 'number') {
      const parts: string[] = [];
      if (opt.min !== undefined) parts.push(`>= ${opt.min}`);
      if (opt.max !== undefined) parts.push(`<= ${opt.max}`);
      if (parts.length > 0) {
        desc += ` (${parts.join(', ')})`;
      }
    }

    // Add default
    if ('default' in opt && opt.default !== undefined) {
      const defaultStr =
        typeof opt.default === 'string' ? `"${opt.default}"` : String(opt.default);
      desc += ` (default: ${defaultStr})`;
    }

    optionStrings.push({ option: optionStr, description: desc });
  }

  // Add help option
  optionStrings.push({ option: '-h, --help', description: 'Show this help message' });

  // Find max option length
  const maxLen = Math.max(...optionStrings.map((o) => o.option.length));

  // Format lines
  for (const { option, description } of optionStrings) {
    lines.push(`  ${option.padEnd(maxLen + 2)}${description}`);
  }

  return lines.join('\n');
}

/**
 * Parse and validate command line arguments against a schema
 *
 * @example
 * ```ts
 * const args = parseArgs({
 *   prompt: { type: 'string', default: 'Hello', description: 'The prompt' },
 *   maxTokens: { type: 'number', default: 256, min: 1, description: 'Max tokens' },
 *   verbose: { type: 'boolean', alias: 'v', description: 'Verbose output' },
 * } as const);
 * ```
 */
export function parseArgs<T extends ArgSchema>(schema: T): ParsedArgs<T> {
  const rawArgs = getArguments();
  const helpText = generateHelpText(schema);

  // Check for help flag first - automatically send help
  const helpRequested = rawArgs.includes('-h') || rawArgs.includes('--help');
  if (helpRequested) {
    send(helpText);
    // Return early with defaults - no validation errors for help
    const defaults: Record<string, unknown> = {};
    for (const [key, opt] of Object.entries(schema)) {
      if ('default' in opt && opt.default !== undefined) {
        defaults[key] = opt.default;
      } else if (opt.type === 'boolean') {
        defaults[key] = false;
      } else if (opt.type === 'string') {
        defaults[key] = '';
      } else if (opt.type === 'number') {
        // Use min value if specified, otherwise 0
        const numOpt = opt as { min?: number };
        defaults[key] = numOpt.min ?? 0;
      }
    }
    return { ...defaults, help: true, helpText } as ParsedArgs<T>;
  }

  // Build lookup maps - support both kebab-case and camelCase
  const keyByKebab = new Map<string, string>();
  const keyByAlias = new Map<string, string>();

  for (const [key, opt] of Object.entries(schema)) {
    keyByKebab.set(toKebabCase(key), key);  // --max-tokens -> maxTokens
    keyByKebab.set(key, key);                // --maxTokens -> maxTokens (also allow camelCase)
    if (opt.alias) {
      keyByAlias.set(opt.alias, key);
    }
  }

  // Parse arguments
  const parsed: Record<string, unknown> = {};
  let i = 0;

  while (i < rawArgs.length) {
    const arg = rawArgs[i];

    // Handle --option=value format
    if (arg.startsWith('--') && arg.includes('=')) {
      const eqIndex = arg.indexOf('=');
      const kebabName = arg.slice(2, eqIndex);
      const value = arg.slice(eqIndex + 1);
      const key = keyByKebab.get(kebabName);

      if (key) {
        parsed[key] = value;
      }
      i++;
      continue;
    }

    // Handle --option value format
    if (arg.startsWith('--')) {
      const kebabName = arg.slice(2);

      // Handle --no-option for booleans
      if (kebabName.startsWith('no-')) {
        const actualKebab = kebabName.slice(3);
        const key = keyByKebab.get(actualKebab);
        if (key && schema[key].type === 'boolean') {
          parsed[key] = false;
          i++;
          continue;
        }
      }

      const key = keyByKebab.get(kebabName);
      if (key) {
        const opt = schema[key];
        if (opt.type === 'boolean') {
          parsed[key] = true;
        } else if (i + 1 < rawArgs.length) {
          parsed[key] = rawArgs[++i];
        }
      }
      i++;
      continue;
    }

    // Handle -a value or -a (for boolean)
    if (arg.startsWith('-') && arg.length === 2) {
      const alias = arg[1];
      const key = keyByAlias.get(alias);

      if (key) {
        const opt = schema[key];
        if (opt.type === 'boolean') {
          parsed[key] = true;
        } else if (i + 1 < rawArgs.length) {
          parsed[key] = rawArgs[++i];
        }
      }
      i++;
      continue;
    }

    i++;
  }

  // Validate and convert types
  const result: Record<string, unknown> = {};

  for (const [key, opt] of Object.entries(schema)) {
    const kebabName = toKebabCase(key);
    let value = parsed[key];

    // Apply default if not provided
    if (value === undefined) {
      if ('default' in opt && opt.default !== undefined) {
        result[key] = opt.default;
        continue;
      } else if (opt.type === 'boolean') {
        result[key] = false;
        continue;
      } else {
        // Required but not provided
        throw new Error(`Error: --${kebabName} is required\n\n${helpText}`);
      }
    }

    // Type conversion and validation
    if (opt.type === 'string') {
      const strValue = String(value);

      // Check choices
      if (opt.choices && !opt.choices.includes(strValue)) {
        throw new Error(
          `Error: --${kebabName}: must be one of [${opt.choices.join(', ')}], got '${strValue}'\n\n${helpText}`
        );
      }

      result[key] = strValue;
    } else if (opt.type === 'number') {
      const numValue = Number(value);

      if (!Number.isFinite(numValue)) {
        throw new Error(
          `Error: --${kebabName}: expected number, got '${value}'\n\n${helpText}`
        );
      }

      // Check min
      if (opt.min !== undefined && numValue < opt.min) {
        throw new Error(
          `Error: --${kebabName}: must be >= ${opt.min}, got ${numValue}\n\n${helpText}`
        );
      }

      // Check max
      if (opt.max !== undefined && numValue > opt.max) {
        throw new Error(
          `Error: --${kebabName}: must be <= ${opt.max}, got ${numValue}\n\n${helpText}`
        );
      }

      // Check choices
      if (opt.choices && !opt.choices.includes(numValue)) {
        throw new Error(
          `Error: --${kebabName}: must be one of [${opt.choices.join(', ')}], got ${numValue}\n\n${helpText}`
        );
      }

      result[key] = numValue;
    } else if (opt.type === 'boolean') {
      // Already handled as true/false during parsing
      result[key] = value === true || value === 'true';
    }
  }

  return { ...result, help: false, helpText } as ParsedArgs<T>;
}
