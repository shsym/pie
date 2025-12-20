// Text Completion Example - JavaScript/TypeScript Inferlet
// Demonstrates basic text generation using the inferlet library

import { Context, Sampler, getAutoModel, getArguments, send } from 'inferlet';

// Get parsed arguments
const args = getArguments();
const prompt = (args.prompt as string) ?? 'Hello, world!';
const maxTokens = Number(args.maxTokens ?? 256);
const system = (args.system as string) ?? 'You are a helpful, respectful and honest assistant.';

// Get the model
const model = getAutoModel();

// Create a context for generation
const ctx = new Context(model);

// Fill the context with messages
ctx.fillSystem(system);
ctx.fillUser(prompt);

// Generate the response
const result = await ctx.generate({
  sampling: Sampler.topP(0.6, 0.95),
  stop: { maxTokens, sequences: model.eosTokens }
});

// Send the result
send(result);
