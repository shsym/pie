wit_bindgen::generate!({
    path: "../../api/wit",
    world: "app",
    pub_export_macro: true,
    export_macro_name: "export",
    with: {
         "wasi:io/poll@0.2.4": wasi::io::poll,
    },
    generate_all,
});

struct HelloWorld;

impl exports::symphony::nbi::run::Guest for HelloWorld {
    fn run() -> Result<(), String> {
        println!("Hello World!!");

        let inst_id = symphony::nbi::runtime::get_instance_id();
        let version = symphony::nbi::runtime::get_version();
        println!(
            "I am an instance (id: {}) running in the Symphony runtime (version: {}) !",
            inst_id, version
        );
        Ok(())
    }
}

export!(HelloWorld);
