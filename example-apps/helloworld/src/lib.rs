use symphony::Run;

struct HelloWorld;

impl Run for HelloWorld {
    async fn run() -> Result<(), String> {
        let inst_id = symphony::system::get_instance_id();
        let version = symphony::system::get_version();
        println!(
            "[{inst_id}] I am a WASM module running in the Symphony ({version}) runtime!",
            inst_id = inst_id,
            version = version
        );

        symphony::system::send_to_origin("Hello world!!");
        symphony::system::send_to_origin("Have a great day!");

        Ok(())
    }
}

symphony::main!(HelloWorld);
