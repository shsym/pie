// Text Completion Example - JavaScript/TypeScript Inferlet
// Demonstrates basic text generation using the inferlet library
// No boilerplate needed - just write top-level code!

const HELP = `
Usage: text-completion-js [OPTIONS]

A simple text completion example for the Pie runtime.

Options:
  --prompt=<text>      The prompt to complete (default: "Hello, world!")
  --max-tokens=<n>     Maximum tokens to generate (default: 256)
                       Must be a positive integer (> 0)
  --system=<text>      System prompt (default: helpful assistant)
  -h, --help           Prints this help message
`;

// Parse command line arguments
// Supports both --option=value and --option value formats
function parseArgs(args: string[]): {
  help: boolean;
  prompt: string;
  maxTokens: number;
  system: string;
} {
  let help = false;
  let prompt = 'Hello, world!';
  let maxTokens = 256;
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
    } else if (arg.startsWith('--system=')) {
      system = arg.slice('--system='.length);
    } else if (arg === '--system' && i + 1 < args.length) {
      system = args[++i];
    }
  }

  return { help, prompt, maxTokens, system };
}

// Main logic - top-level await!
const args = getArguments();
const { help, prompt: userPrompt, maxTokens, system } = parseArgs(args);

if (help) {
  send(HELP);
} else {
  // Validate numeric arguments
  if (!Number.isFinite(maxTokens) || !Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error(
      `Invalid --max-tokens value: must be a positive integer (> 0), got '${maxTokens}'`
    );
  }

  // Get the model
  const model = getAutoModel();

  // Create a context for generation
  const ctx = new Context(model);

  // Use ChatFormatter for proper prompt formatting
  ctx.fillSystem(system);
  ctx.fillUser(userPrompt);

  // Create sampler and stop condition
  const sampler = Sampler.topP(0.6, 0.95);
  const eosTokens = model.eosTokens().map((arr) => [...arr]);
  const stopCond = maxLen(maxTokens).or(endsWithAny(eosTokens));

  // Generate the response
  const result = await ctx.generate(sampler, stopCond);

  // Send the result
  send(result);
  send('\n');
}
