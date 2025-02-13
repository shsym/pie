extern crate prost_build;

fn main() {
    prost_build::compile_protos(&["../backend-api/sys.proto"], &["../backend-api"]).unwrap();
}
