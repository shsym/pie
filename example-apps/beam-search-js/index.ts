// Beam Search Example - JavaScript/TypeScript Inferlet
// Demonstrates beam search decoding for text generation using the inferlet library

import {
  getAutoModel,
  getArguments,
  send,
  Context,
  maxLen,
  endsWithAny,
} from 'inferlet';

const HELP = `
Usage: beam-search-js [OPTIONS]

A program to run text generation with beam search decoding.

Options:
  --prompt <text>      The prompt to send to the model
                       [default: Explain the LLM decoding process ELI5.]
  --max-tokens <n>     Maximum tokens to generate (default: 128)
                       Must be a positive integer (> 0)
  --beam-size <n>      The beam size for decoding (default: 1)
                       Must be a positive integer (>= 1)
  --system <text>      System prompt (default: helpful assistant)
  -h, --help           Prints this help message
`;

// Parse command line arguments
// Supports both --option=value and --option value formats
function parseArgs(args: string[]): {
  help: boolean;
  prompt: string;
  maxTokens: number;
  beamSize: number;
  system: string;
} {
  let help = false;
  let prompt = 'Explain the LLM decoding process ELI5.';
  let maxTokens = 128;
  let beamSize = 1;
  let system = 'You are a helpful, respectful and honest assistant.';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-h' || arg === '--help') {
      help = true;
    } else if (arg.startsWith('--prompt=')) {
      prompt = arg.slice('--prompt='.length);
    } else if (arg === '--prompt' && i + 1 < args.length) {
      prompt = args[++i];
    } else if (arg.startsWith('--max-tokens=')) {
      maxTokens = parseInt(arg.slice('--max-tokens='.length), 10);
    } else if (arg === '--max-tokens' && i + 1 < args.length) {
      maxTokens = parseInt(args[++i], 10);
    } else if (arg.startsWith('--beam-size=')) {
      beamSize = parseInt(arg.slice('--beam-size='.length), 10);
    } else if (arg === '--beam-size' && i + 1 < args.length) {
      beamSize = parseInt(args[++i], 10);
    } else if (arg.startsWith('--system=')) {
      system = arg.slice('--system='.length);
    } else if (arg === '--system' && i + 1 < args.length) {
      system = args[++i];
    }
  }

  return { help, prompt, maxTokens, beamSize, system };
}

// Main implementation
async function main(): Promise<void> {
  const args = getArguments();
  const { help, prompt, maxTokens, beamSize, system } = parseArgs(args);

  if (help) {
    send(HELP);
    return;
  }

  // Validate numeric arguments
  if (!Number.isFinite(maxTokens) || !Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error(
      `Invalid --max-tokens value: must be a positive integer (> 0), got '${maxTokens}'`
    );
  }
  if (!Number.isFinite(beamSize) || !Number.isInteger(beamSize) || beamSize < 1) {
    throw new Error(
      `Invalid --beam-size value: must be a positive integer (>= 1), got '${beamSize}'`
    );
  }

  // Get the model
  const model = getAutoModel();

  // Create a context for generation
  const ctx = new Context(model);

  // Format prompt in Llama 3 style
  const formattedPrompt = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>

${system}<|eot_id|><|start_header_id|>user<|end_header_id|>

${prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

`;

  ctx.fill(formattedPrompt);

  // Create stop condition
  const eosTokens = model.eosTokens().map((arr) => [...arr]);
  const stopCond = maxLen(maxTokens).or(endsWithAny(eosTokens));

  // Generate the response using beam search
  const result = await ctx.generateWithBeam(stopCond, beamSize);

  // Send the result
  send(result);
  send('\n');
}

// Export in WIT-compatible format for inferlet:core/run interface
export const run = {
  run: async (): Promise<{ tag: 'ok' } | { tag: 'err'; val: string }> => {
    try {
      await main();
      return { tag: 'ok' };
    } catch (e) {
      return { tag: 'err', val: String(e) };
    }
  },
};
