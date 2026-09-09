use alloc::string::String;

const PROLOGUE: &str = include_str!("../../../runtime/cuda/ptir_m1_runtime_prologue.cuh");
const BODY: &str = include_str!("../../../runtime/cuda/ptir_m1_runtime_body.cuh");

fn rng_preamble() -> String {
    let mut preamble = String::from("\n");
    preamble.push_str(&crate::codegen::rng::cuda_device_functions());
    preamble
}

pub fn singleton_runtime_source() -> String {
    let mut source = String::with_capacity(PROLOGUE.len() + BODY.len() + 4096);
    source.push_str(PROLOGUE);
    source.push_str(&rng_preamble());
    source.push_str(BODY);
    source
}
