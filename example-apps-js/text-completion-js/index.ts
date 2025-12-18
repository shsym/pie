// Text Completion Example - JavaScript/TypeScript Inferlet
// Demonstrates basic text generation using the inferlet library

import { Context, Sampler, getAutoModel, parseArgs, send } from 'inferlet';

// Parse args - automatically sends help text if --help is passed
const args = parseArgs({
  prompt: {
    type: 'string',
    default: 'Hello, world!',
    description: 'The prompt to complete'
  },
  maxTokens: {
    type: 'number',
    default: 256,
    min: 1,
    description: 'Maximum tokens to generate'
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

  // Generate the response using the new object-based API
  const result = await ctx.generate({
    messages: [
      { role: 'system', content: args.system },
      { role: 'user', content: args.prompt }
    ],
    sampling: Sampler.topP(0.6, 0.95),
    stop: { maxTokens: args.maxTokens, sequences: model.eosTokens }
  });

  // Send the result
  send(result);
}
