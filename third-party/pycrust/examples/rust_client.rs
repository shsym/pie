//! Example Rust client for PyCrust.
//!
//! To run:
//!     1. Start the Python worker: python examples/python_worker.py
//!     2. Run this client: cargo run --example rust_client

use pycrust_client::RpcClient;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct AddRequest {
    a: i32,
    b: i32,
}

#[derive(Serialize)]
struct MultiplyRequest {
    x: f64,
    y: f64,
}

#[derive(Serialize)]
struct ProcessDataRequest {
    data: Vec<i32>,
    operation: String,
}

#[derive(Deserialize, Debug)]
struct ProcessDataResponse {
    result: f64,
    operation: String,
    count: usize,
}

#[derive(Serialize)]
struct EchoRequest {
    message: String,
}

#[derive(Serialize)]
struct FibonacciRequest {
    n: i32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Connecting to calculator service...");
    let client = RpcClient::connect("calculator").await?;

    // Test ping
    println!("\n--- Testing ping ---");
    let pong: String = client.call("ping", &()).await?;
    println!("Ping response: {}", pong);

    // Test add
    println!("\n--- Testing add ---");
    let result: i32 = client.call("add", &AddRequest { a: 10, b: 20 }).await?;
    println!("10 + 20 = {}", result);

    // Test multiply
    println!("\n--- Testing multiply ---");
    let result: f64 = client
        .call("multiply", &MultiplyRequest { x: 3.14, y: 2.0 })
        .await?;
    println!("3.14 * 2.0 = {}", result);

    // Test process_data with different operations
    println!("\n--- Testing process_data ---");
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    for operation in ["sum", "mean", "max", "min"] {
        let result: ProcessDataResponse = client
            .call(
                "process_data",
                &ProcessDataRequest {
                    data: data.clone(),
                    operation: operation.to_string(),
                },
            )
            .await?;
        println!("  {} of {:?}: {:?}", operation, data, result);
    }

    // Test echo
    println!("\n--- Testing echo ---");
    let result: String = client.call("echo", &"Hello, PyCrust!").await?;
    println!("Echo: {}", result);

    // Test fibonacci
    println!("\n--- Testing fibonacci ---");
    for n in [10, 20, 30, 40] {
        let result: i64 = client.call("fibonacci", &n).await?;
        println!("fib({}) = {}", n, result);
    }

    // Test with timeout
    println!("\n--- Testing with timeout ---");
    let result: String = client
        .call_with_timeout("ping", &(), std::time::Duration::from_secs(5))
        .await?;
    println!("Ping with timeout: {}", result);

    println!("\nAll tests passed!");
    client.close().await;
    Ok(())
}
