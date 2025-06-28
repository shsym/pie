extern crate prost_build;
use flatbuffers_build::BuilderOptions;

fn main() {
    prost_build::compile_protos(
        &[
            "../api/backend/handshake.proto",
            "../api/backend/ping.proto",
            "../api/backend/l4m.proto",
            "../api/backend/l4m_vision.proto",
        ],
        &["../api/backend/", "../api/frontend/"],
    )
    .unwrap();

    BuilderOptions::new_with_files([
        "../api/backend/handshake.fbs",
        "../api/backend/gpt.fbs",
        "../api/backend/gpt_vision.fbs",
        "../api/backend/ping.fbs",
        "../api/backend/main.fbs"
    ])
    .compile()
    .expect("flatbuffer compilation failed");
}
