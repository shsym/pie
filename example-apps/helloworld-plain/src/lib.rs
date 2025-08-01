wit_bindgen::generate!({
    path: "../../inferlet/wit",
    world: "inferlet",
    pub_export_macro: true,
    export_macro_name: "export",
    with: {
         "wasi:io/poll@0.2.4": wasi::io::poll,
    },
    generate_all,
});

struct HelloWorld;
impl exports::pie::inferlet::run::Guest for HelloWorld {
    fn run() -> Result<(), String> {
        println!("Hello World!!");

        let inst_id = pie::inferlet::core::get_instance_id();
        let version = pie::inferlet::core::get_version();
        println!(
            "I am an instance (id: {}) running in the PIE runtime (version: {}) !",
            inst_id, version
        );
        Ok(())
    }
}

export!(HelloWorld);
