# Bakery

**Pie Bakery** is a CLI tool for building and publishing inferlets. It supports both JavaScript/TypeScript and Rust inferlets.

## Installation

```bash
pip install -e sdk/tools/bakery
```

## Commands

### `bakery create`

Create a new inferlet project.

```bash
# Create a Rust inferlet (default)
bakery create my-inferlet

# Create a TypeScript inferlet
bakery create my-inferlet --ts
```

### `bakery build`

Build an inferlet to WebAssembly. The platform (Rust or JavaScript) is auto-detected based on project files.

```bash
# Build a Rust inferlet (directory with Cargo.toml)
bakery build ./my-rust-inferlet -o output.wasm

# Build a TypeScript inferlet (directory with package.json)
bakery build ./my-ts-inferlet -o output.wasm

# Build a single JS/TS file
bakery build ./index.ts -o output.wasm
```

**Options:**
- `-o, --output` - Output `.wasm` file path (required)
- `--debug` - Enable debug build (JS only: includes source maps)

### `bakery login`

Authenticate with the Pie Registry using GitHub OAuth.

```bash
bakery login
```

### `bakery inferlet`

Manage inferlets in the Pie Registry.

```bash
# Search for inferlets
bakery inferlet search <query>

# Get inferlet info
bakery inferlet info <name>

# Publish an inferlet
bakery inferlet publish <path>
```

## Requirements

### For Rust inferlets:
- Rust toolchain with `cargo`
- wasm32-wasip2 target: `rustup target add wasm32-wasip2`

### For JavaScript/TypeScript inferlets:
- Node.js v18+
- npm/npx

## Development

When developing within the pie repository, set `PIE_SDK` to the SDK directory:

```bash
export PIE_SDK=/path/to/pie/sdk
```

This allows bakery to find the inferlet SDK libraries automatically.
