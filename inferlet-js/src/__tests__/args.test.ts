import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { parseArgs } from '../args.js';
import * as runtime from '../runtime.js';

// Mock the runtime module
vi.mock('../runtime.js', () => ({
  getArguments: vi.fn(() => [])
}));

// Mock the messaging module (has WIT imports that don't exist in tests)
vi.mock('../messaging.js', () => ({
  send: vi.fn()
}));

describe('parseArgs', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('help flag', () => {
    it('should set help to true when -h is passed', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['-h']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'test', description: 'Test prompt' }
      } as const);

      expect(args.help).toBe(true);
      expect(args.helpText).toBeTruthy();
    });

    it('should set help to true when --help is passed', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'test', description: 'Test prompt' }
      } as const);

      expect(args.help).toBe(true);
      expect(args.helpText).toBeTruthy();
    });

    it('should populate defaults when help is requested', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'Hello', description: 'The prompt' },
        maxTokens: { type: 'number', default: 256, description: 'Max tokens' },
        verbose: { type: 'boolean', description: 'Verbose output' }
      } as const);

      expect(args.help).toBe(true);
      expect(args.prompt).toBe('Hello');
      expect(args.maxTokens).toBe(256);
      expect(args.verbose).toBe(false);
    });
  });

  describe('help text generation', () => {
    it('should generate help text with option descriptions', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'Hello', description: 'The prompt' },
        maxTokens: { type: 'number', default: 256, description: 'Max tokens' }
      } as const);

      expect(args.helpText).toContain('Usage:');
      expect(args.helpText).toContain('--prompt');
      expect(args.helpText).toContain('--max-tokens');
      expect(args.helpText).toContain('The prompt');
      expect(args.helpText).toContain('Max tokens');
      expect(args.helpText).toContain('--help');
    });

    it('should show aliases in help text', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        verbose: { type: 'boolean', alias: 'v', description: 'Verbose output' }
      } as const);

      expect(args.helpText).toContain('-v, --verbose');
    });

    it('should show default values in help text', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'Hello', description: 'Test' },
        count: { type: 'number', default: 42, description: 'Number' }
      } as const);

      expect(args.helpText).toContain('default: "Hello"');
      expect(args.helpText).toContain('default: 42');
    });

    it('should show choices in help text', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        mode: {
          type: 'string',
          choices: ['fast', 'slow', 'auto'] as const,
          default: 'auto',
          description: 'Mode'
        }
      } as const);

      expect(args.helpText).toContain('[fast|slow|auto]');
    });

    it('should show min/max ranges in help text', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        tokens: {
          type: 'number',
          default: 100,
          min: 1,
          max: 1000,
          description: 'Token count'
        }
      } as const);

      expect(args.helpText).toContain('>= 1');
      expect(args.helpText).toContain('<= 1000');
    });

    it('should convert camelCase to kebab-case in help text', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        maxTokens: { type: 'number', default: 100, description: 'Max tokens' },
        beamSize: { type: 'number', default: 1, description: 'Beam size' }
      } as const);

      expect(args.helpText).toContain('--max-tokens');
      expect(args.helpText).toContain('--beam-size');
    });
  });

  describe('string options', () => {
    it('should parse string option with --option=value format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--prompt=Hello World']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('Hello World');
    });

    it('should parse string option with --option value format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--prompt', 'Hello World']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('Hello World');
    });

    it('should parse string option with alias -a value format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['-p', 'Hello']);

      const args = parseArgs({
        prompt: { type: 'string', alias: 'p', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('Hello');
    });

    it('should use default value when string option is not provided', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('default');
    });

    it('should throw error when required string option is not provided', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      expect(() => {
        parseArgs({
          prompt: { type: 'string', description: 'Required prompt' }
        } as const);
      }).toThrow('--prompt is required');
    });

    it('should validate string choices', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--mode=invalid']);

      expect(() => {
        parseArgs({
          mode: {
            type: 'string',
            choices: ['fast', 'slow'] as const,
            default: 'fast',
            description: 'Mode'
          }
        } as const);
      }).toThrow('must be one of [fast, slow]');
    });

    it('should accept valid string choices', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--mode=slow']);

      const args = parseArgs({
        mode: {
          type: 'string',
          choices: ['fast', 'slow'] as const,
          default: 'fast',
          description: 'Mode'
        }
      } as const);

      expect(args.mode).toBe('slow');
    });

    it('should handle kebab-case conversion for camelCase keys', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--max-tokens=100']);

      const args = parseArgs({
        maxTokens: { type: 'string', default: '50', description: 'Test' }
      } as const);

      expect(args.maxTokens).toBe('100');
    });
  });

  describe('number options', () => {
    it('should parse number option with --option=value format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=42']);

      const args = parseArgs({
        count: { type: 'number', default: 10, description: 'Test' }
      } as const);

      expect(args.count).toBe(42);
    });

    it('should parse number option with --option value format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count', '42']);

      const args = parseArgs({
        count: { type: 'number', default: 10, description: 'Test' }
      } as const);

      expect(args.count).toBe(42);
    });

    it('should parse number option with alias', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['-c', '42']);

      const args = parseArgs({
        count: { type: 'number', alias: 'c', default: 10, description: 'Test' }
      } as const);

      expect(args.count).toBe(42);
    });

    it('should use default value when number option is not provided', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      const args = parseArgs({
        count: { type: 'number', default: 10, description: 'Test' }
      } as const);

      expect(args.count).toBe(10);
    });

    it('should throw error when required number option is not provided', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      expect(() => {
        parseArgs({
          count: { type: 'number', description: 'Required count' }
        } as const);
      }).toThrow('--count is required');
    });

    it('should throw error for invalid number value', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=invalid']);

      expect(() => {
        parseArgs({
          count: { type: 'number', default: 10, description: 'Test' }
        } as const);
      }).toThrow('expected number');
    });

    it('should validate minimum value', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=5']);

      expect(() => {
        parseArgs({
          count: { type: 'number', default: 20, min: 10, description: 'Test' }
        } as const);
      }).toThrow('must be >= 10');
    });

    it('should validate maximum value', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=150']);

      expect(() => {
        parseArgs({
          count: { type: 'number', default: 50, max: 100, description: 'Test' }
        } as const);
      }).toThrow('must be <= 100');
    });

    it('should accept value within min/max range', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=50']);

      const args = parseArgs({
        count: { type: 'number', default: 20, min: 10, max: 100, description: 'Test' }
      } as const);

      expect(args.count).toBe(50);
    });

    it('should validate number choices', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--size=15']);

      expect(() => {
        parseArgs({
          size: {
            type: 'number',
            choices: [1, 5, 10] as const,
            default: 1,
            description: 'Size'
          }
        } as const);
      }).toThrow('must be one of [1, 5, 10]');
    });

    it('should accept valid number choices', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--size=5']);

      const args = parseArgs({
        size: {
          type: 'number',
          choices: [1, 5, 10] as const,
          default: 1,
          description: 'Size'
        }
      } as const);

      expect(args.size).toBe(5);
    });

    it('should parse floating point numbers', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--temp=0.75']);

      const args = parseArgs({
        temp: { type: 'number', default: 1.0, description: 'Temperature' }
      } as const);

      expect(args.temp).toBe(0.75);
    });

    it('should parse negative numbers', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--offset=-10']);

      const args = parseArgs({
        offset: { type: 'number', default: 0, description: 'Offset' }
      } as const);

      expect(args.offset).toBe(-10);
    });
  });

  describe('boolean options', () => {
    it('should parse boolean flag with --option format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--verbose']);

      const args = parseArgs({
        verbose: { type: 'boolean', description: 'Verbose output' }
      } as const);

      expect(args.verbose).toBe(true);
    });

    it('should parse boolean flag with alias -a format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['-v']);

      const args = parseArgs({
        verbose: { type: 'boolean', alias: 'v', description: 'Verbose output' }
      } as const);

      expect(args.verbose).toBe(true);
    });

    it('should parse negated boolean flag with --no-option format', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--no-verbose']);

      const args = parseArgs({
        verbose: { type: 'boolean', default: true, description: 'Verbose output' }
      } as const);

      expect(args.verbose).toBe(false);
    });

    it('should default to false when boolean option is not provided', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      const args = parseArgs({
        verbose: { type: 'boolean', description: 'Verbose output' }
      } as const);

      expect(args.verbose).toBe(false);
    });

    it('should use custom default value for boolean', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      const args = parseArgs({
        verbose: { type: 'boolean', default: true, description: 'Verbose output' }
      } as const);

      expect(args.verbose).toBe(true);
    });

    it('should handle kebab-case conversion for camelCase boolean keys', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--dry-run']);

      const args = parseArgs({
        dryRun: { type: 'boolean', description: 'Dry run mode' }
      } as const);

      expect(args.dryRun).toBe(true);
    });
  });

  describe('mixed options', () => {
    it('should parse multiple options of different types', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([
        '--prompt',
        'Hello',
        '--max-tokens=100',
        '-v'
      ]);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Prompt' },
        maxTokens: { type: 'number', default: 50, description: 'Max tokens' },
        verbose: { type: 'boolean', alias: 'v', description: 'Verbose' }
      } as const);

      expect(args.prompt).toBe('Hello');
      expect(args.maxTokens).toBe(100);
      expect(args.verbose).toBe(true);
    });

    it('should handle options in any order', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([
        '-v',
        '--max-tokens',
        '200',
        '--prompt=Test'
      ]);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Prompt' },
        maxTokens: { type: 'number', default: 50, description: 'Max tokens' },
        verbose: { type: 'boolean', alias: 'v', description: 'Verbose' }
      } as const);

      expect(args.prompt).toBe('Test');
      expect(args.maxTokens).toBe(200);
      expect(args.verbose).toBe(true);
    });

    it('should use defaults for unprovided options', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--prompt=Test']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Prompt' },
        maxTokens: { type: 'number', default: 50, description: 'Max tokens' },
        verbose: { type: 'boolean', description: 'Verbose' }
      } as const);

      expect(args.prompt).toBe('Test');
      expect(args.maxTokens).toBe(50);
      expect(args.verbose).toBe(false);
    });
  });

  describe('complex real-world scenarios', () => {
    it('should parse text-completion-js style arguments', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([
        '--prompt=Hello, world!',
        '--max-tokens=256',
        '--system=You are helpful'
      ]);

      const args = parseArgs({
        prompt: { type: 'string', default: 'Hello, world!', description: 'The prompt' },
        maxTokens: {
          type: 'number',
          default: 256,
          min: 1,
          description: 'Max tokens to generate'
        },
        system: {
          type: 'string',
          default: 'You are a helpful assistant',
          description: 'System prompt'
        }
      } as const);

      expect(args.prompt).toBe('Hello, world!');
      expect(args.maxTokens).toBe(256);
      expect(args.system).toBe('You are helpful');
    });

    it('should parse beam-search-js style arguments', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([
        '--prompt=Explain AI',
        '--max-tokens=128',
        '--beam-size=4'
      ]);

      const args = parseArgs({
        prompt: {
          type: 'string',
          default: 'Explain the LLM decoding process ELI5.',
          description: 'The prompt'
        },
        maxTokens: {
          type: 'number',
          default: 128,
          min: 1,
          description: 'Max tokens'
        },
        beamSize: {
          type: 'number',
          default: 1,
          min: 1,
          description: 'Beam size'
        },
        system: {
          type: 'string',
          default: 'You are a helpful assistant',
          description: 'System prompt'
        }
      } as const);

      expect(args.prompt).toBe('Explain AI');
      expect(args.maxTokens).toBe(128);
      expect(args.beamSize).toBe(4);
      expect(args.system).toBe('You are a helpful assistant');
    });

    it('should handle empty argument list', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      const args = parseArgs({
        prompt: { type: 'string', default: 'test', description: 'Test' },
        count: { type: 'number', default: 10, description: 'Count' },
        verbose: { type: 'boolean', description: 'Verbose' }
      } as const);

      expect(args.prompt).toBe('test');
      expect(args.count).toBe(10);
      expect(args.verbose).toBe(false);
      expect(args.help).toBe(false);
    });

    it('should handle unknown options gracefully', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([
        '--prompt=test',
        '--unknown=value',
        '--another-unknown'
      ]);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      // Unknown options should be ignored
      expect(args.prompt).toBe('test');
    });
  });

  describe('type inference', () => {
    it('should infer correct types from schema', () => {
      vi.mocked(runtime.getArguments).mockReturnValue([]);

      const args = parseArgs({
        str: { type: 'string', default: 'test', description: 'String' },
        num: { type: 'number', default: 42, description: 'Number' },
        bool: { type: 'boolean', description: 'Boolean' },
        choice: {
          type: 'string',
          choices: ['a', 'b', 'c'] as const,
          default: 'a',
          description: 'Choice'
        }
      } as const);

      // Type assertions to verify type inference
      const str: string = args.str;
      const num: number = args.num;
      const bool: boolean = args.bool;
      const choice: 'a' | 'b' | 'c' = args.choice;
      const help: boolean = args.help;
      const helpText: string = args.helpText;

      expect(str).toBe('test');
      expect(num).toBe(42);
      expect(bool).toBe(false);
      expect(choice).toBe('a');
      expect(help).toBe(false);
      expect(helpText).toBeTruthy();
    });
  });

  describe('edge cases', () => {
    it('should handle option values that look like other options', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--prompt=--test']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('--test');
    });

    it('should handle empty string values', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--prompt=']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('');
    });

    it('should handle zero as a valid number', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=0']);

      const args = parseArgs({
        count: { type: 'number', default: 10, description: 'Test' }
      } as const);

      expect(args.count).toBe(0);
    });

    it('should handle very large numbers', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--count=999999999']);

      const args = parseArgs({
        count: { type: 'number', default: 10, description: 'Test' }
      } as const);

      expect(args.count).toBe(999999999);
    });

    it('should handle scientific notation', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--value=1e6']);

      const args = parseArgs({
        value: { type: 'number', default: 0, description: 'Test' }
      } as const);

      expect(args.value).toBe(1000000);
    });

    it('should handle values with special characters', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--prompt=Hello, world! How are you?']);

      const args = parseArgs({
        prompt: { type: 'string', default: 'default', description: 'Test' }
      } as const);

      expect(args.prompt).toBe('Hello, world! How are you?');
    });
  });

  describe('help defaults with min constraint', () => {
    it('should use min value for numbers without explicit default on help', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        count: {
          type: 'number',
          min: 1,
          description: 'Count value'
        }
      } as const);

      expect(args.help).toBe(true);
      expect(args.count).toBe(1);  // Should use min, not 0
    });

    it('should prefer explicit default over min on help', () => {
      vi.mocked(runtime.getArguments).mockReturnValue(['--help']);

      const args = parseArgs({
        count: {
          type: 'number',
          default: 10,
          min: 1,
          description: 'Count value'
        }
      } as const);

      expect(args.help).toBe(true);
      expect(args.count).toBe(10);  // Should use explicit default
    });
  });
});
