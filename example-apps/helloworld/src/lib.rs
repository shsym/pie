use symphony::Run;

struct HelloWorld;

impl Run for HelloWorld {
    async fn run() -> Result<(), String> {
        let inst_id = symphony::runtime::get_instance_id();
        let version = symphony::runtime::get_runtime_version();
        println!(
            "[{inst_id}] I am a WASM module running in the Symphony ({version}) runtime!",
            inst_id = inst_id,
            version = version
        );

        symphony::messaging::broadcast(inst_id, "Hello world!!");
        symphony::messaging::send_to_origin("Have a great day!");

        Ok(())
    }
}

symphony::main!(HelloWorld);
