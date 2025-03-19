#[symphony::main]
async fn main() -> Result<(), String> {
    println!("Hello World!!");

    let inst_id = symphony::runtime::get_instance_id();
    let version = symphony::runtime::get_version();
    println!(
        "I am an instance (id: {}) running in the Symphony runtime (version: {}) !",
        inst_id, version
    );
    Ok(())
}
