//! PyCrust End-to-End Benchmark
//!
//! This benchmark measures the full IPC round-trip latency and throughput
//! between a Rust client and Python worker.
//!
//! To run:
//!     1. Start the Python worker: python examples/benchmark_worker.py
//!     2. Run this benchmark: cargo run --example rust_benchmark --release

use pycrust_client::RpcClient;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ==============================================================================
// Request/Response Types
// ==============================================================================

#[derive(Serialize)]
struct Empty {}

#[derive(Serialize)]
struct AddRequest {
    a: i32,
    b: i32,
}

#[derive(Serialize)]
struct ForwardPassRequest {
    input_tokens: Vec<i32>,
    input_token_positions: Vec<i32>,
    adapter: Option<i32>,
    adapter_seed: Option<i32>,
    mask: Vec<Vec<i32>>,
    kv_page_ptrs: Vec<i64>,
    kv_page_last_len: i32,
    output_token_indices: Vec<i32>,
}

#[derive(Deserialize, Debug)]
struct ForwardPassResponse {
    tokens: Vec<i32>,
    dists: Vec<(Vec<i32>, Vec<f64>)>,
    num_processed: usize,
}

#[derive(Serialize)]
struct HandshakeRequest {
    version: String,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct HandshakeResponse {
    version: String,
    model_name: String,
    model_traits: Vec<String>,
    kv_page_size: i32,
    max_batch_tokens: i32,
}

#[derive(Serialize)]
struct BatchProcessRequest {
    items: Vec<BatchItem>,
    operation: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct BatchItem {
    id: i32,
    value: i32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct BatchProcessResponse {
    result: Option<f64>,
    count: usize,
    items: Option<Vec<BatchItem>>,
}

#[derive(Serialize)]
struct EchoRequest {
    data: Vec<u8>,
}

#[derive(Deserialize, Debug)]
struct EchoResponse {
    data: Vec<u8>,
    size: usize,
}

#[derive(Serialize)]
struct ComputeHeavyRequest {
    iterations: i32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ComputeHeavyResponse {
    result: i64,
    iterations: i32,
}

#[derive(Serialize)]
struct NestedDataRequest {
    depth: i32,
    width: i32,
}

// ==============================================================================
// Benchmark Infrastructure
// ==============================================================================

#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    iterations: usize,
    total_time_ms: f64,
    mean_latency_us: f64,
    median_latency_us: f64,
    p99_latency_us: f64,
    min_latency_us: f64,
    max_latency_us: f64,
    throughput_ops_sec: f64,
}

fn calculate_stats(latencies: &mut [f64]) -> (f64, f64, f64, f64, f64) {
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let median = latencies[latencies.len() / 2];
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99 = latencies[p99_idx.min(latencies.len() - 1)];
    let min = latencies[0];
    let max = latencies[latencies.len() - 1];

    (mean, median, p99, min, max)
}

fn print_result(result: &BenchmarkResult) {
    println!("\n  {}:", result.name);
    println!("    Iterations:     {}", result.iterations);
    println!("    Total time:     {:.2} ms", result.total_time_ms);
    println!("    Mean latency:   {:.2} us", result.mean_latency_us);
    println!("    Median latency: {:.2} us", result.median_latency_us);
    println!("    P99 latency:    {:.2} us", result.p99_latency_us);
    println!("    Min latency:    {:.2} us", result.min_latency_us);
    println!("    Max latency:    {:.2} us", result.max_latency_us);
    println!("    Throughput:     {:.0} ops/sec", result.throughput_ops_sec);
}

// ==============================================================================
// Individual Benchmarks
// ==============================================================================

async fn benchmark_noop(client: &RpcClient, iterations: usize) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let call_start = Instant::now();
        let _: () = client.call("noop", &Empty {}).await.unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_time = start.elapsed();

    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: "noop (minimal overhead)".to_string(),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

async fn benchmark_add(client: &RpcClient, iterations: usize) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    let start = Instant::now();
    for i in 0..iterations {
        let call_start = Instant::now();
        let _: i32 = client
            .call("add", &AddRequest { a: i as i32, b: 42 })
            .await
            .unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_time = start.elapsed();

    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: "add (simple args)".to_string(),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

async fn benchmark_forward_pass(client: &RpcClient, iterations: usize) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    // Prepare a realistic forward pass request
    let seq_len = 128;
    let request = ForwardPassRequest {
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
        let call_start = Instant::now();
        let _: ForwardPassResponse = client.call("forward_pass", &request).await.unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_time = start.elapsed();

    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: "forward_pass (ML inference simulation)".to_string(),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

async fn benchmark_batch_process(
    client: &RpcClient,
    iterations: usize,
    batch_size: usize,
) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    // Prepare batch data
    let items: Vec<BatchItem> = (0..batch_size)
        .map(|i| BatchItem {
            id: i as i32,
            value: i as i32 * 2,
        })
        .collect();

    let request = BatchProcessRequest {
        items,
        operation: "sum".to_string(),
    };

    let start = Instant::now();
    for _ in 0..iterations {
        let call_start = Instant::now();
        let _: BatchProcessResponse = client.call("batch_process", &request).await.unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_time = start.elapsed();

    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: format!("batch_process ({} items)", batch_size),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

async fn benchmark_echo(
    client: &RpcClient,
    iterations: usize,
    payload_size: usize,
) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    // Prepare payload
    let data: Vec<u8> = (0..payload_size).map(|i| (i % 256) as u8).collect();
    let request = EchoRequest { data };

    let start = Instant::now();
    for _ in 0..iterations {
        let call_start = Instant::now();
        let _: EchoResponse = client.call("echo", &request).await.unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_time = start.elapsed();

    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: format!("echo ({} bytes)", payload_size),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

async fn benchmark_throughput(client: &RpcClient, duration_secs: f64) -> BenchmarkResult {
    let start = Instant::now();
    let end_time = Duration::from_secs_f64(duration_secs);
    let mut iterations = 0;
    let mut latencies = Vec::new();

    while start.elapsed() < end_time {
        let call_start = Instant::now();
        let _: i32 = client.call("add", &AddRequest { a: 1, b: 2 }).await.unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
        iterations += 1;
    }

    let total_time = start.elapsed();
    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: format!("sustained throughput ({:.0}s)", duration_secs),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

// ==============================================================================
// Concurrent Benchmarks
// ==============================================================================

async fn benchmark_concurrent_throughput(
    client: &std::sync::Arc<RpcClient>,
    concurrency: usize,
    duration_secs: f64,
) -> BenchmarkResult {
    let start = Instant::now();
    let end_time = Duration::from_secs_f64(duration_secs);
    let mut tasks = Vec::with_capacity(concurrency);

    // Spawn concurrent tasks
    for _ in 0..concurrency {
        let client = client.clone();
        let end_time_clone = start + end_time; // Approx end time

        tasks.push(tokio::spawn(async move {
            let mut local_latencies = Vec::new();
            let mut local_iterations = 0;

            while Instant::now() < end_time_clone {
                let call_start = Instant::now();
                // Use a simple add call for throughput
                let _: i32 = client
                    .call("add", &AddRequest { a: 1, b: 2 })
                    .await
                    .unwrap();
                local_latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
                local_iterations += 1;
            }
            (local_iterations, local_latencies)
        }));
    }

    // Collect results
    let mut total_iterations = 0;
    let mut all_latencies = Vec::new();

    for task in tasks {
        let (iters, mut lats) = task.await.unwrap();
        total_iterations += iters;
        all_latencies.append(&mut lats);
    }

    let total_time = start.elapsed();
    let (mean, median, p99, min, max) = calculate_stats(&mut all_latencies);

    BenchmarkResult {
        name: format!("concurrent throughput ({} tasks, {:.0}s)", concurrency, duration_secs),
        iterations: total_iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: total_iterations as f64 / total_time.as_secs_f64(),
    }
}

async fn benchmark_concurrent_correctness(
    client: &std::sync::Arc<RpcClient>,
    concurrency: usize,
    iterations_per_task: usize,
) {
    let mut tasks = Vec::with_capacity(concurrency);

    println!(
        "  Running correctness check with {} tasks, {} iterations each...",
        concurrency, iterations_per_task
    );

    for task_id in 0..concurrency {
        let client = client.clone();
        tasks.push(tokio::spawn(async move {
            for i in 0..iterations_per_task {
                // Use the 'add' function with specific values to verify the result matches THIS request
                // request: a = task_id, b = i
                // expected: task_id + i
                let a = task_id as i32;
                let b = i as i32;
                let expected = a + b;

                let result: i32 = client.call("add", &AddRequest { a, b }).await.unwrap();

                if result != expected {
                    panic!(
                        "Correctness failure in task {}: add({}, {}) returned {}, expected {}",
                        task_id, a, b, result, expected
                    );
                }
            }
        }));
    }

    for task in tasks {
        task.await.unwrap();
    }
    println!("  PASSED");
}

async fn benchmark_notify(client: &RpcClient, iterations: usize) -> BenchmarkResult {
    let mut latencies = Vec::with_capacity(iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let call_start = Instant::now();
        client.notify("noop", &Empty {}).unwrap();
        latencies.push(call_start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_time = start.elapsed();

    let (mean, median, p99, min, max) = calculate_stats(&mut latencies);

    BenchmarkResult {
        name: "notify (fire-and-forget)".to_string(),
        iterations,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        mean_latency_us: mean,
        median_latency_us: median,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        throughput_ops_sec: iterations as f64 / total_time.as_secs_f64(),
    }
}

// ==============================================================================
// Main Benchmark Runner
// ==============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!(" PyCrust End-to-End Benchmark");
    println!("============================================================");
    println!("\nConnecting to benchmark service...");

    let client = std::sync::Arc::new(RpcClient::connect("benchmark_v5").await?);

    // Verify connection with handshake
    let handshake: HandshakeResponse = client
        .call("handshake", &HandshakeRequest { version: "1.0.0".to_string() })
        .await?;
    println!("Connected to: {} v{}", handshake.model_name, handshake.version);

    // Warmup
    println!("\nWarming up...");
    for _ in 0..1000 {
        let _: () = client.call("noop", &Empty {}).await?;
    }

    println!("\n============================================================");
    println!(" Latency Benchmarks (10,000 iterations each)");
    println!("============================================================");
    println!("(Running sequentially to measure pure latency)");

    // Run latency benchmarks
    let iterations = 10000;

    let result = benchmark_noop(&client, iterations).await;
    print_result(&result);

    let result = benchmark_notify(&client, iterations).await;
    print_result(&result);

    let result = benchmark_add(&client, iterations).await;
    print_result(&result);

    let result = benchmark_forward_pass(&client, iterations).await;
    print_result(&result);

    println!("\n============================================================");
    println!(" Payload Size Benchmarks (5,000 iterations each)");
    println!("============================================================");

    let iterations = 5000;

    for payload_size in [100, 1000, 10000, 30000] {
        let result = benchmark_echo(&client, iterations, payload_size).await;
        print_result(&result);
    }

    println!("\n============================================================");
    println!(" Batch Size Benchmarks (5,000 iterations each)");
    println!("============================================================");

    for batch_size in [10, 100, 1000] {
        let result = benchmark_batch_process(&client, iterations, batch_size).await;
        print_result(&result);
    }

    println!("\n============================================================");
    println!(" Sequential Throughput (5 second sustained load)");
    println!("============================================================");

    let result = benchmark_throughput(&client, 5.0).await;
    print_result(&result);

    println!("\n============================================================");
    println!(" Concurrent Throughput (3 second sustained load)");
    println!("============================================================");

    for concurrency in [2, 4, 8, 16, 32, 64, 128] {
        let result = benchmark_concurrent_throughput(&client, concurrency, 3.0).await;
        print_result(&result);
    }

    println!("\n============================================================");
    println!(" Concurrent Correctness Check");
    println!("============================================================");

    benchmark_concurrent_correctness(&client, 4, 1000).await;
    benchmark_concurrent_correctness(&client, 32, 1000).await;
    benchmark_concurrent_correctness(&client, 128, 500).await;

    println!("\n============================================================");
    println!(" Summary");
    println!("============================================================");

    // Run final measurements for summary
    let noop_result = benchmark_noop(&client, 1000).await;
    let notify_result = benchmark_notify(&client, 1000).await;
    let add_result = benchmark_add(&client, 1000).await;
    
    // Quick re-run of max concurrency for summary
    let concurrent_result = benchmark_concurrent_throughput(&client, 32, 1.0).await;

    println!("\n  Baseline (noop) median latency:  {:.2} us", noop_result.median_latency_us);
    println!("  Notify (fire-forget) latency:    {:.2} us (client side only)", notify_result.median_latency_us);
    println!("  Simple call median latency:      {:.2} us", add_result.median_latency_us);
    println!("  Max Concurrent Throughput:       {:.0} ops/sec", concurrent_result.throughput_ops_sec);

    println!("\n============================================================");
    println!(" Benchmark Complete");
    println!("============================================================\n");

    if let Ok(c) = std::sync::Arc::try_unwrap(client) {
         c.close().await;
    } else {
         println!("(Could not unwrap Arc to close client explicitly, dropping instead)");
    }
    
    Ok(())
}
