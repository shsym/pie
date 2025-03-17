extern crate prost_build;

fn main() {
    prost_build::compile_protos(
        &[
            "../api/backend/handshake.proto",
            "../api/backend/ping.proto",
            "../api/backend/l4m.proto",
            "../api/backend/l4m_vision.proto",
            "../api/frontend/client.proto",
        ],
        &["../api/backend/", "../api/frontend/"],
    )
    .unwrap();
}
