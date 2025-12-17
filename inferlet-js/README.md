# inferlet-js

JavaScript/TypeScript library for writing Pie inferlets.

## Quick Start with Examples

See `example-apps-js/` for complete examples:

- **text-completion-js** - Basic text generation with sampling
- **beam-search-js** - Beam search decoding

Build and run an example:

```bash
cd example-apps-js/text-completion-js
pie-cli build . -o text-completion.wasm
pie-cli submit text-completion.wasm -- --prompt "Hello"
```

## How to Create My Own Inferlet

```bash
# Create a new inferlet with TypeScript (recommended)
pie-cli create my-inferlet
```

For JavaScript, use `pie-cli create my-inferlet --js`.

This generates:
- `index.ts` - Your inferlet code (or `index.js` for JavaScript)
- `package.json` - Package manifest
- `tsconfig.json` - TypeScript configuration with path mappings (TypeScript only)

### Build

```bash
cd my-inferlet
pie-cli build . -o my-inferlet.wasm
```

### Run

Make sure the Pie server is running:

```bash
pie-cli ping
```

Then submit the compiled inferlet:

```bash
pie-cli submit my-inferlet.wasm
```

## Writing Inferlets

Inferlets use **top-level await**. Write your logic directly without boilerplate:

```typescript
// my-inferlet/index.ts

const model = getAutoModel();
const ctx = new Context(model);

ctx.fillSystem('You are a helpful assistant.');
ctx.fillUser('Hello!');

const sampler = Sampler.topP(0.6, 0.95);
const eosTokens = model.eosTokens().map((arr) => [...arr]);
const stopCond = maxLen(256).or(endsWithAny(eosTokens));

const result = await ctx.generate(sampler, stopCond);
send(result);
send('\n');
```

The build system automatically:
- Injects all inferlet globals (`getAutoModel`, `Context`, `Sampler`, etc.)
- Wraps your code in the WIT interface
- Handles error reporting

## Available Globals

### Core
- `getAutoModel()` - Returns the model instance
- `getArguments()` - Returns command-line arguments as an array
- `send(text)` - Sends output to the client

### Classes
- `Context` - Generation context with KV cache
- `Sampler` - Token sampling strategies (`.greedy()`, `.topP()`, `.topK()`)
- `ChatFormatter` - Chat template formatting
- `Tokenizer` - Text tokenization

### Stop Conditions
- `maxLen(n)` - Stops after `n` tokens
- `endsWith(tokens)` - Stops when output ends with the specified tokens
- `endsWithAny(tokenArrays)` - Stops when output ends with any of the token sequences

Stop conditions can be combined: `maxLen(256).or(endsWithAny(eosTokens))`

## CLI Reference

### Create

```bash
pie-cli create <name> [OPTIONS]

Options:
  --js               Use JavaScript instead of TypeScript
  -o, --output <dir> Output directory (default: current directory)
```

### Build

```bash
pie-cli build <input> -o <output.wasm> [OPTIONS]

Options:
  --debug    Use debug build of StarlingMonkey runtime
```

## TypeScript Support

The generated `tsconfig.json` provides full IDE support:
- Auto-completion for all inferlet APIs
- Type checking for your code
- No explicit imports needed

Path mappings point to `inferlet-js/src/` for type definitions.
