#[pie::main]
async fn main() -> Result<(), String> {
    println!("Hello World!!");

    let inst_id = pie::runtime::get_instance_id();
    let version = pie::runtime::get_version();
    println!(
        "I am an instance (id: {}) running in the Symphony runtime (version: {}) !",
        inst_id, version
    );
    Ok(())
}
