use symphony::Run;

struct HelloWorld;

impl Run for HelloWorld {
    async fn run() -> Result<(), String> {
        let inst_id = symphony::runtime::get_instance_id();
        let version = symphony::runtime::get_version();
        println!(
            "[{inst_id}] I am a WASM module running in the Symphony ({version}) runtime!",
            inst_id = inst_id,
            version = version
        );

        symphony::messaging::send("Hello world!!");
        symphony::messaging::send("Have a great day!");

        Ok(())
    }
}

symphony::main!(HelloWorld);
