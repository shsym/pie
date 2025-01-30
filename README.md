# Symphony

## Getting Started

**1. Install the WASI Preview 2 Target**
```bash
rustup target add wasm32-wasip2
```

**2. Compile the Example LIP App**  
The Rust example app (`helloworld`) is in `./example-apps/helloworld`.
```bash
cd ./example-apps/helloworld
cargo build --target wasm32-wasip2 --release
```
This uses `wit-bindgen` to implement the `spi:app/run` interface. The compiled `helloworld.wasm` file is located in `./example-apps/target/wasm32-wasip2/release/`.

**3. Compile Symphony**  
Symphony currently lacks an LLM backend, but it can still serve LIP apps (LLM-related API calls are stubbed).
```bash
cd ./engine
cargo build
cargo run
```

**4. Test the Example App**  
Use the `symphony-toolkit` in `./toolkit` to run and interact with the `helloworld` LIP app. See `./toolkit/README.md` for details.