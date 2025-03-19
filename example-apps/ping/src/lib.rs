use std::time::{Duration, Instant};

#[symphony::main]
async fn main() -> Result<(), String> {
    const PING_COUNT: usize = 30;
    let mut total_duration = Duration::ZERO;

    println!("Starting ping test with {} iterations...", PING_COUNT);

    for i in 1..=PING_COUNT {
        let start = Instant::now();
        let resp = symphony::ping::ping("hello");
        let elapsed = start.elapsed();
        total_duration += elapsed;

        println!("Ping {}: Response: {:?}, Latency: {:?}", i, resp, elapsed);
    }

    let avg_latency = total_duration / PING_COUNT as u32;
    println!("\nPing test completed.");
    println!("Total Pings: {}", PING_COUNT);
    println!("Average Latency: {:?}", avg_latency);

    Ok(())
}
