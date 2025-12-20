// Beam Search Example - JavaScript/TypeScript Inferlet
// Demonstrates beam search decoding for text generation

import { Context, getAutoModel, getArguments, send, maxLen, endsWithAny } from 'inferlet';

// Get parsed arguments
const args = getArguments();
const prompt = (args.prompt as string) ?? 'Explain the LLM decoding process ELI5.';
const maxTokens = Number(args.maxTokens ?? 128);
const beamSize = Number(args.beamSize ?? 1);
const system = (args.system as string) ?? 'You are a helpful, respectful and honest assistant.';

// Get the model
const model = getAutoModel();

// Create a context for generation
const ctx = new Context(model);

// Use ChatFormatter for proper prompt formatting (model-agnostic)
ctx.fillSystem(system);
ctx.fillUser(prompt);

// Create stop condition (beam search still uses StopCondition API)
const stopCond = maxLen(maxTokens).or(endsWithAny(model.eosTokens));

// Generate the response using beam search
const result = await ctx.generateWithBeam(stopCond, beamSize);

// Send the result
send(result);
