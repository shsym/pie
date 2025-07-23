extern crate prost_build;

fn main() {
    prost_build::compile_protos(
        &[
            "../backend/proto/handshake.proto",
            "../backend/proto/ping.proto",
            "../backend/proto/l4m.proto",
            "../backend/proto/l4m_vision.proto",
        ],
        &["../backend/proto"],
    )
    .unwrap();

}
