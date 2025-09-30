use inferlet::wstd::time::Instant;
use pico_args::Arguments;
use std::ffi::OsString;

const HELP: &str = r#"
Usage: program [OPTIONS]

An inferlet for benchmarking the execution latency.

Options:
  -i, --index <INTEGER>          An optional integer index for storing latency.
  -l, --layer <STRING>           The layer to send the message to. (either 'control' or 'inference')
                                 (default: "control")
  -a, --aggregate-size <UINT>    If set, aggregates latencies from the store instead of measuring.
  -h, --help                     Print help information.
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // 1. Get arguments from the inferlet environment and prepare the parser.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // --- Print help information if requested ---
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // --- Parse all arguments ---
    let index: Option<u32> = args
        .opt_value_from_str(["-i", "--index"])
        .map_err(|e| e.to_string())?;

    let layer = args
        .opt_value_from_str(["-l", "--layer"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "control".to_string());

    // New argument: aggregate_size
    let aggregate_size: Option<u32> = args
        .opt_value_from_str(["-a", "--aggregate-size"])
        .map_err(|e| e.to_string())?;

    // --- Main Logic ---

    // If aggregate_size is set, perform aggregation and exit.
    if let Some(size) = aggregate_size {
        let mut latencies: Vec<u128> = Vec::with_capacity(size as usize);
        println!("Aggregating up to {} latencies...", size);

        for i in 0..size {
            let key = format!("latency-{}", i);
            if let Some(value_str) = inferlet::store_get(&key) {
                // Parse the string value to u128 and add it to the vector.
                match value_str.parse::<u128>() {
                    Ok(latency) => latencies.push(latency),
                    Err(_) => eprintln!("Warning: Could not parse value for key '{}'", key),
                }
            } else {
                eprintln!("Warning: No value found for key '{}'", key);
            }
        }

        let count = latencies.len();
        if count == 0 {
            println!("No valid latencies found to aggregate.");
            return Ok(());
        }

        // --- Calculate Mean ---
        let sum: u128 = latencies.iter().sum();
        let mean = sum as f64 / count as f64;

        // --- Calculate Median ---
        latencies.sort_unstable();
        let median = if count % 2 == 0 {
            let mid_idx = count / 2;
            (latencies[mid_idx - 1] + latencies[mid_idx]) as f64 / 2.0
        } else {
            latencies[count / 2] as f64
        };

        // --- Calculate Standard Deviation ---
        let variance = latencies
            .iter()
            .map(|value| {
                let diff = *value as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        println!("\nAggregation Results (from {} samples):", count);
        println!("  - Mean:   {:.2} µs", mean);
        println!("  - Median: {:.2} µs", median);
        println!("  - StdDev: {:.2} µs", std_dev);

        let json = format!(
            r#"{{"samples":{},"mean":{:.2},"median":{:.2},"std_dev":{:.2}}}"#,
            count, mean, median, std_dev
        );

        inferlet::send(&json);

        return Ok(());
    }

    // If aggregate_size is not set, perform performance measurement.
    if layer != "control" && layer != "inference" {
        return Err(format!("Invalid layer: {}", layer));
    }

    if layer == "control" {
        let start_time = Instant::now();
        let _res = inferlet::debug_query("ping").await; // handled in the control layer
        let elapsed = start_time.elapsed();

        println!("Execution latency: {} µs", elapsed.as_micros());

        // store the result in kvs
        let key = format!("latency-{}", index.unwrap_or(0));
        let value = elapsed.as_micros().to_string();
        inferlet::store_set(&key, &value);

        return Ok(());
    }

    if layer == "inference" {
        let model = inferlet::get_auto_model();
        let start_time = Instant::now();
        let _res = model.create_queue().debug_query("ping").await; // handled in the inference layer
        let elapsed = start_time.elapsed();

        println!("Execution latency: {} µs", elapsed.as_micros());

        // store the result in kvs
        let key = format!("latency-{}", index.unwrap_or(0));
        let value = elapsed.as_micros().to_string();
        inferlet::store_set(&key, &value);

        return Ok(());
    }

    Ok(())
}
