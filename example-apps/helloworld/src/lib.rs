wit_bindgen::generate!({
    path: "../../api/app/wit",
    world: "app",
    generate_all,
});

use crate::exports::spi::app::run::Guest;
use crate::spi::app::system;

struct HelloWorld;

impl Guest for HelloWorld {
    fn run() -> Result<(), ()> {
        println!("I am a WASM module running in the Symphony runtime!");

        system::send_to_origin("What is your name?");
        system::send_to_origin("Have a great day!");
        Ok(())
    }
}

export!(HelloWorld);
