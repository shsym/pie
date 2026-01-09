//! Rust-side Latency Breakdown for PyCrust
//!
//! This measures:
//! 1. MessagePack serialization/deserialization
//! 2. mpsc channel round-trip
//! 3. iceoryx2 raw overhead (publish/subscribe)
//!
//! Run: cargo run --example rust_latency_breakdown --release

use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Clone)]
struct SimplePayload {
    a: i32,
    b: i32,
}

#[derive(Serialize, Deserialize, Clone)]
struct ComplexPayload {
    input_tokens: Vec<i32>,
    input_token_positions: Vec<i32>,
    adapter: Option<i32>,
    adapter_seed: Option<i32>,
    mask: Vec<Vec<i32>>,
    kv_page_ptrs: Vec<i64>,
    kv_page_last_len: i32,
    output_token_indices: Vec<i32>,
}

fn measure_msgpack_serialize_simple(iterations: usize) -> f64 {
    let payload = SimplePayload { a: 1, b: 2 };

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rmp_serde::to_vec_named(&payload).unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_msgpack_deserialize_simple(iterations: usize) -> f64 {
    let payload = SimplePayload { a: 1, b: 2 };
    let encoded = rmp_serde::to_vec_named(&payload).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _: SimplePayload = rmp_serde::from_slice(&encoded).unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_msgpack_roundtrip_simple(iterations: usize) -> f64 {
    let payload = SimplePayload { a: 1, b: 2 };

    let start = Instant::now();
    for _ in 0..iterations {
        let encoded = rmp_serde::to_vec_named(&payload).unwrap();
        let _: SimplePayload = rmp_serde::from_slice(&encoded).unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_msgpack_roundtrip_complex(iterations: usize) -> f64 {
    let seq_len = 128;
    let payload = ComplexPayload {
        input_tokens: (0..seq_len).collect(),
        input_token_positions: (0..seq_len).collect(),
        adapter: Some(1),
        adapter_seed: Some(42),
        mask: vec![vec![1; seq_len as usize]; seq_len as usize],
        kv_page_ptrs: vec![0x1000, 0x2000, 0x3000, 0x4000],
        kv_page_last_len: 64,
        output_token_indices: vec![seq_len - 1],
    };

    let start = Instant::now();
    for _ in 0..iterations {
        let encoded = rmp_serde::to_vec_named(&payload).unwrap();
        let _: ComplexPayload = rmp_serde::from_slice(&encoded).unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_mpsc_channel_roundtrip(iterations: usize) -> f64 {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();

    let payload = rmp_serde::to_vec_named(&SimplePayload { a: 1, b: 2 }).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        tx.send(payload.clone()).unwrap();
        let _ = rx.recv().unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_mpsc_try_recv_empty(iterations: usize) -> f64 {
    let (_tx, rx) = mpsc::channel::<Vec<u8>>();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rx.try_recv();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_thread_sleep_50us(iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        std::thread::sleep(Duration::from_micros(50));
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_thread_sleep_100us(iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        std::thread::sleep(Duration::from_micros(100));
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_iceoryx_roundtrip(iterations: usize) -> Result<f64, String> {
    use iceoryx2::prelude::*;

    // Create a unique service name
    let service_name = format!("latency_test_{}", std::process::id());

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| format!("Failed to create node: {}", e))?;

    let service = node
        .service_builder(&service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| format!("Failed to create service: {}", e))?;

    let publisher = service
        .publisher_builder()
        .initial_max_slice_len(1024)
        .create()
        .map_err(|e| format!("Failed to create publisher: {}", e))?;

    let subscriber = service
        .subscriber_builder()
        .create()
        .map_err(|e| format!("Failed to create subscriber: {}", e))?;

    let payload = rmp_serde::to_vec_named(&SimplePayload { a: 1, b: 2 }).unwrap();

    // Warmup
    for _ in 0..100 {
        let sample = publisher.loan_slice_uninit(payload.len()).unwrap();
        let sample = sample.write_from_slice(&payload);
        sample.send().unwrap();

        // Poll for response
        loop {
            if let Ok(Some(_)) = subscriber.receive() {
                break;
            }
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let sample = publisher.loan_slice_uninit(payload.len()).unwrap();
        let sample = sample.write_from_slice(&payload);
        sample.send().unwrap();

        // Poll for response (busy wait)
        loop {
            if let Ok(Some(_)) = subscriber.receive() {
                break;
            }
        }
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn measure_iceoryx_publish_only(iterations: usize) -> Result<f64, String> {
    use iceoryx2::prelude::*;

    let service_name = format!("latency_pub_{}", std::process::id());

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| format!("Failed to create node: {}", e))?;

    let service = node
        .service_builder(&service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| format!("Failed to create service: {}", e))?;

    let publisher = service
        .publisher_builder()
        .initial_max_slice_len(1024)
        .create()
        .map_err(|e| format!("Failed to create publisher: {}", e))?;

    let payload = rmp_serde::to_vec_named(&SimplePayload { a: 1, b: 2 }).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let sample = publisher.loan_slice_uninit(payload.len()).unwrap();
        let sample = sample.write_from_slice(&payload);
        sample.send().unwrap();
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn main() {
    println!("============================================================");
    println!(" PyCrust Rust-Side Latency Breakdown");
    println!("============================================================");

    let iterations = 100000;
    println!("\nMeasuring with {} iterations each...\n", iterations);

    println!("MessagePack Serialization:");
    println!("----------------------------------------");
    let ser = measure_msgpack_serialize_simple(iterations);
    println!("  Simple serialize:         {:.3} µs", ser);

    let de = measure_msgpack_deserialize_simple(iterations);
    println!("  Simple deserialize:       {:.3} µs", de);

    let rt = measure_msgpack_roundtrip_simple(iterations);
    println!("  Simple round-trip:        {:.3} µs", rt);

    let complex_rt = measure_msgpack_roundtrip_complex(iterations / 10);
    println!("  Complex round-trip:       {:.3} µs", complex_rt);

    println!("\nChannel Communication:");
    println!("----------------------------------------");
    let mpsc_rt = measure_mpsc_channel_roundtrip(iterations);
    println!("  mpsc channel round-trip:  {:.3} µs", mpsc_rt);

    let mpsc_try = measure_mpsc_try_recv_empty(iterations);
    println!("  mpsc try_recv (empty):    {:.3} µs", mpsc_try);

    println!("\nPolling Sleep Overhead:");
    println!("----------------------------------------");
    let sleep_50 = measure_thread_sleep_50us(1000);
    println!("  sleep(50µs) actual:       {:.1} µs", sleep_50);

    let sleep_100 = measure_thread_sleep_100us(1000);
    println!("  sleep(100µs) actual:      {:.1} µs", sleep_100);

    println!("\niceoryx2 IPC:");
    println!("----------------------------------------");
    match measure_iceoryx_publish_only(iterations) {
        Ok(pub_only) => println!("  Publish only:             {:.3} µs", pub_only),
        Err(e) => println!("  Publish only:             ERROR: {}", e),
    }

    match measure_iceoryx_roundtrip(iterations / 10) {
        Ok(ipc_rt) => println!("  Loopback (same process):  {:.3} µs", ipc_rt),
        Err(e) => println!("  Loopback:                 ERROR: {}", e),
    }

    println!("\n============================================================");
    println!(" Summary");
    println!("============================================================");
    println!();
    println!("Estimated overhead per RPC round-trip:");
    println!();
    println!("  Rust-side (client):");
    println!("    - Request serialize:     ~{:.1} µs", ser);
    println!("    - Response deserialize:  ~{:.1} µs", de);
    println!("    - mpsc channel:          ~{:.1} µs x2", mpsc_rt);
    println!("    - Transport poll sleep:  ~{:.1} µs (worst case)", sleep_50);
    println!();
    println!("  Python-side (worker):");
    println!("    - Request deserialize:   ~{:.1} µs", de);
    println!("    - Response serialize:    ~{:.1} µs", ser);
    println!("    - Worker poll sleep:     ~{:.1} µs (worst case)", sleep_100);
    println!();
    println!("  CRITICAL FINDING:");
    println!("    The 50µs + 100µs poll sleeps cause ~150µs worst-case latency!");
    println!("    This explains the ~300µs observed baseline latency.");
    println!();
}
