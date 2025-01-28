
wit_bindgen::generate!({
    world: "app",
    generate_all,
});

struct HelloWorld;
use crate::exports::spi::core::run::Guest;

impl Guest for HelloWorld {
    fn run() -> Result<(), ()> {
        spi::core::system::ask("What is your name?");
        spi::core::system::tell("Have a great day!");
        Ok(())
    }
}

export!(HelloWorld);
