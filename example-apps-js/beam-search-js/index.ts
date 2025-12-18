// Beam Search Example - JavaScript/TypeScript Inferlet
// Demonstrates beam search decoding for text generation

import { Context, getAutoModel, parseArgs, send, maxLen, endsWithAny } from 'inferlet';

// Parse args - automatically sends help text if --help is passed
const args = parseArgs({
  prompt: {
    type: 'string',
    default: 'Explain the LLM decoding process ELI5.',
    description: 'The prompt to send to the model'
  },
  maxTokens: {
    type: 'number',
    default: 128,
    min: 1,
    description: 'Maximum tokens to generate'
  },
  beamSize: {
    type: 'number',
    default: 1,
    min: 1,
    description: 'The beam size for decoding'
  },
  system: {
    type: 'string',
    default: 'You are a helpful, respectful and honest assistant.',
    description: 'System prompt'
  }
} as const);

// Only run main logic if help was not requested
if (!args.help) {
  // Get the model
  const model = getAutoModel();

  // Create a context for generation
  const ctx = new Context(model);

  // Use ChatFormatter for proper prompt formatting (model-agnostic)
  ctx.fillSystem(args.system);
  ctx.fillUser(args.prompt);

  // Create stop condition (beam search still uses StopCondition API)
  const stopCond = maxLen(args.maxTokens).or(endsWithAny(model.eosTokens));

  // Generate the response using beam search
  const result = await ctx.generateWithBeam(stopCond, args.beamSize);

  // Send the result
  send(result);
}
