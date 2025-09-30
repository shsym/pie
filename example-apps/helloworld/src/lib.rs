#[inferlet::main]
async fn main() -> Result<(), String> {
    println!("Hello World!!");

    let inst_id = inferlet::get_instance_id();
    let version = inferlet::get_version();
    println!(
        "I am an instance (id: {}) running in the PIE runtime (version: {}) !",
        inst_id, version
    );
    Ok(())
}
