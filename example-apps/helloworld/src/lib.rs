use symphony::Run;

struct App;

impl Run for App {
    async fn run() -> Result<(), String> {
        let inst_id = symphony::system::get_instance_id();

        println!(
            "[{}] I am a WASM module running in the Symphony runtime!",
            { inst_id }
        );

        symphony::system::send_to_origin("What is your name?");
        symphony::system::send_to_origin("Have a great day!");

        Ok(())
    }
}

symphony::main!(App);
