//! PyCrust FFI Benchmark
//!
//! Run with: cargo run --release --example benchmark

use pycrust_ffi::FfiClient;
use pyo3::prelude::*;
use pyo3::ffi::c_str;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Serialize)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Serialize)]
struct EchoArgs {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
}

#[derive(Deserialize)]
struct EchoResponse {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
}

fn main() -> PyResult<()> {
    println!("============================================================");
    println!(" PyCrust FFI Benchmark");
    println!("============================================================\n");
    
    Python::with_gil(|py| {
        // Create worker and register methods
        let worker_code = c_str!(r#"
import msgpack

STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INTERNAL_ERROR = 3

methods = {}

def noop():
    return None

def add(a, b):
    return a + b

def echo(data):
    return {"data": data}

methods["noop"] = noop
methods["add"] = add
methods["echo"] = echo

def dispatch(method, payload):
    fn = methods.get(method)
    if fn is None:
        return (STATUS_METHOD_NOT_FOUND, msgpack.packb(f"Method not found: {method}"))
    try:
        args = msgpack.unpackb(payload)
        if isinstance(args, dict):
            result = fn(**args)
        elif isinstance(args, (list, tuple)):
            result = fn(*args)
        else:
            result = fn()
        return (STATUS_OK, msgpack.packb(result))
    except Exception as e:
        return (STATUS_INTERNAL_ERROR, msgpack.packb(str(e)))
"#);
        
        // Execute Python code to define dispatch function
        py.run(worker_code, None, None)?;
        let dispatch = py.eval(c_str!("dispatch"), None, None)?;
        
        // Create FFI client
        let client = FfiClient::new(dispatch.into());
        
        println!("FFI client created. Running benchmarks...\n");
        
        // Warmup
        for _ in 0..1000 {
            let _: Option<()> = client.call(py, "noop", &()).unwrap();
        }
        
        // =================================================================
        // Latency Benchmarks
        // =================================================================
        println!("============================================================");
        println!(" Latency Benchmarks (10,000 iterations each)");
        println!("============================================================\n");
        
        // noop benchmark
        let iterations = 10_000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _: Option<()> = client.call(py, "noop", &()).unwrap();
        }
        let elapsed = start.elapsed();
        print_stats("noop", iterations, elapsed);
        
        // add benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _: i32 = client.call(py, "add", &AddArgs { a: 10, b: 20 }).unwrap();
        }
        let elapsed = start.elapsed();
        print_stats("add", iterations, elapsed);
        
        // notify benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            client.notify(py, "noop", &()).unwrap();
        }
        let elapsed = start.elapsed();
        print_stats("notify", iterations, elapsed);
        
        // =================================================================
        // Payload Size Benchmarks
        // =================================================================
        println!("\n============================================================");
        println!(" Payload Size Benchmarks (5,000 iterations each)");
        println!("============================================================\n");
        
        for size in [100, 1000, 10000] {
            let data = vec![0u8; size];
            let args = EchoArgs { data };
            let iterations = 5_000;
            
            let start = Instant::now();
            for _ in 0..iterations {
                let _: EchoResponse = client.call(py, "echo", &args).unwrap();
            }
            let elapsed = start.elapsed();
            print_stats(&format!("echo ({} bytes)", size), iterations, elapsed);
        }
        
        // =================================================================
        // Summary
        // =================================================================
        println!("\n============================================================");
        println!(" Benchmark Complete");
        println!("============================================================");
        
        Ok(())
    })
}

fn print_stats(name: &str, iterations: usize, elapsed: Duration) {
    let mean_us = elapsed.as_micros() as f64 / iterations as f64;
    let throughput = iterations as f64 / elapsed.as_secs_f64();
    
    println!("  {}:", name);
    println!("    Iterations:     {}", iterations);
    println!("    Total time:     {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("    Mean latency:   {:.2} us", mean_us);
    println!("    Throughput:     {:.0} ops/sec", throughput);
    println!();
}
