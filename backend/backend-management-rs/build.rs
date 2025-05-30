use std::io::Result;

fn main() -> Result<()> {
    // Build protobuf files
    let mut config = prost_build::Config::new();
    
    // Configure protobuf generation
    config.out_dir("src/proto");
    
    // Build handshake protocol
    config.compile_protos(
        &["../../api/backend/handshake.proto"],
        &["../../api/backend/"]
    )?;
    
    // Build main L4M protocol
    config.compile_protos(
        &["../../api/backend/l4m.proto"],
        &["../../api/backend/"]
    )?;
    
    // Build vision protocol
    config.compile_protos(
        &["../../api/backend/l4m_vision.proto"],
        &["../../api/backend/"]
    )?;
    
    // Build ping protocol
    config.compile_protos(
        &["../../api/backend/ping.proto"],
        &["../../api/backend/"]
    )?;
    
    println!("cargo:rerun-if-changed=../../api/backend/handshake.proto");
    println!("cargo:rerun-if-changed=../../api/backend/l4m.proto");
    println!("cargo:rerun-if-changed=../../api/backend/l4m_vision.proto");
    println!("cargo:rerun-if-changed=../../api/backend/ping.proto");
    
    Ok(())
}
