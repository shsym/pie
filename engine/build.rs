extern crate prost_build;

fn main() {
    prost_build::compile_protos(
        &[
            "../backend-api/handshake.proto",
            "../backend-api/ping.proto",
            "../backend-api/l4m.proto",
            "../backend-api/l4m_vision.proto",
        ],
        &["../backend-api/"],
    )
    .unwrap();
}
