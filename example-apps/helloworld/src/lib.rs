use crate::exports::spi::app::run::Guest;

wit_bindgen::generate!({
    path: "../../spi/app/wit",
    world: "app",
    generate_all,
});

struct HelloWorld;

impl Guest for HelloWorld {
    fn run() -> Result<(), ()> {
        println!("I am a WASM module running in the Symphony runtime!");

        spi::app::system::ask("What is your name?");
        spi::app::system::tell("Have a great day!");
        Ok(())
    }
}

export!(HelloWorld);
