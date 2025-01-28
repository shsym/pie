// src/lib.rs

use wit_bindgen::component::bindgen;

// This macro generates the code to wire up the WIT-defined world.
bindgen!({
    path: "../wit/helloworld.wit",
    world: "../wit/helloworld"
});

// After the macro runs, you'll have generated modules named
// `imports` (for everything in `import { ... }`) and
// `exports` (for everything in `export { ... }`).

// We can bring Spi into scope for calls inside our `main` implementation.
use self::imports::spi as spi_import; // This is where 'tell' and 'ask' live.

// The trait for our exported main function is in `exports::my_component`.
struct Helloworld;

// Implement the exported function(s) for this world.
impl exports::helloworld::Helloworld for Helloworld {
    /// This is our entry point, called "main" in WIT.
    /// It receives a single string parameter: `message`.
    fn main(&mut self, message: String) {
        // Example usage: call the imported `tell` method from `Spi`.
        spi_import::tell(&message);

        // Or if we want to ask a question:
        // let answer = spi_import::ask("How are you?");
        // println!("Response was: {}", answer);
    }
}