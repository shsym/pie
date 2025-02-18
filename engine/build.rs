extern crate prost_build;

fn main() {
    prost_build::compile_protos(&["../backend-api/l4m.proto"], &["../backend-api"]).unwrap();
}
