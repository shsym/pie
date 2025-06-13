fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(
        &["../../api/backend/handshake.proto"],
        &["../../api/backend"],
    )?;
    Ok(())
}
