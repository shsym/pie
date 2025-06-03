# Serving System Engine

This is a standalone software written in Rust

Download the model (Llama 3.2 1B) with the following command. You should have Hugging Face CLI set up and have access to the model with git/ssh.
```bash
./download_tokenizer.sh
```

This will download model files into `program_cache`.

Before running the engine, make sure that you have run the backend with the corresponding model compatible with the tokenizer. You can then run the engine with:

```bash
cargo run --release
```

## Example usage
- OpenAI API compatible server
Note that you first need to run the backend that is compatible with this server. After that, you can run the engine and the server with:
```bash
cargo run --release -- openai_compat -H -p 8080 --dummy false
```

## Testing
- Tokenizer test (metadata and tokenizer loading)
```bash
cargo test --test tokenizer_integration_tests
```

- Demo version of the tokenizer with hand-coded metadata
```bash
cargo run --example tokenizer_demo
```
