# Bakery

**Pie Bakery** is the build tool for inferlets. It supports
**Rust**, **Python**, and **JavaScript / TypeScript** projects.

You normally don't need to install Bakery directly — it ships with
`pie-server`, exposed as `pie build` and `pie new`. The `bakery`
binary is also available standalone for users who don't have
`pie-server` installed.

## Commands

### `pie new` (or `bakery create`)

Create a new inferlet project.

```bash
# Rust (default)
pie new my-inferlet

# TypeScript
pie new my-inferlet --ts
```

### `pie build` (or `bakery build`)

Build an inferlet to a WebAssembly component. The platform is
auto-detected from the project files.

```bash
# Rust         (directory with Cargo.toml)
pie build ./my-rust-inferlet -o out.wasm

# Python       (directory with pyproject.toml or main.py)
pie build ./my-py-inferlet   -o out.wasm

# TypeScript   (directory with package.json)
pie build ./my-ts-inferlet   -o out.wasm

# Single source file
pie build ./index.ts         -o out.wasm
pie build ./main.py          -o out.wasm
```

**Options:**
- `-o, --output` — Output `.wasm` file path (required)
- `--debug` — Enable debug build (Python / JS / TS: includes source maps)

### `bakery login` / `bakery inferlet`

Manage inferlets in the Pie Registry. See `bakery <cmd> --help` for
the GitHub-OAuth login flow and the `inferlet search/info/publish`
subcommands.

## Requirements

| Language       | Toolchain                                                              |
|----------------|------------------------------------------------------------------------|
| **Rust**       | `cargo` + `rustup target add wasm32-wasip2`                            |
| **Python**     | `componentize-py` (pulled in automatically via `factored-componentize-py`); the host needs `~/.pie/py-runtime/` to actually run the WASM — Bakery fetches this on first Python build, or run `pie config init` to do it explicitly |
| **JS / TS**    | Node.js v18+ and `npx`                                                 |

The Python WASM runtime is ~22 MiB and is fetched once. Subsequent
builds reuse the cached copy.

## Errors and conventions

**`raise` / `throw` becomes a WIT `result<_, string>::Err`.** The
auto-generated wrapper in Bakery catches Python exceptions and JS
thrown values, converts them to the WIT `Err` variant, and surfaces
them to the host as the inferlet's error string. So you can write
idiomatic Python `raise Exception("...")` or JS `throw new Error("...")`
and the host will report your message rather than a wasm trap.

## Development

When developing inside the pie repository, Bakery auto-detects the
SDK layout — both from your current working directory and from
Bakery's own install location, so editable installs of Bakery work
out of the box. To override, set:

```bash
export PIE_SDK=/path/to/pie/sdk
```
