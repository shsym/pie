extern crate prost_build;

fn main() {
    prost_build::compile_protos(
        &[
            "proto/handshake.proto",
            "proto/ping.proto",
            "proto/l4m.proto",
            "proto/l4m_vision.proto",
        ],
        &["proto"],
    )
    .unwrap();

}
