use pico_args::Arguments;
use std::ffi::OsString;
use std::fmt;
use std::str::FromStr;

/// Defines the available batching algorithms.
/// The `FromStr` and `Display` traits are implemented for easy parsing
/// from command-line arguments and for formatting into the query string.
#[derive(Debug, PartialEq)]
enum BatchingAlgo {
    TOnly,
    KOnly,
    KOrT,
    Adaptive,
}

/// Implements parsing from a string slice into a `BatchingAlgo` enum.
/// This allows `pico_args` to directly convert the command-line argument
/// into our enum type, handling validation at the same time.
impl FromStr for BatchingAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "t-only" => Ok(BatchingAlgo::TOnly),
            "k-only" => Ok(BatchingAlgo::KOnly),
            "k-or-t" => Ok(BatchingAlgo::KOrT),
            "adaptive" => Ok(BatchingAlgo::Adaptive),
            _ => Err(format!(
                "'{}' is not a valid batching algorithm. Use one of [t-only, k-only, k-or-t, adaptive].",
                s
            )),
        }
    }
}

/// Implements the `Display` trait to convert the enum back into its
/// string representation, which is used when constructing the final query.
impl fmt::Display for BatchingAlgo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BatchingAlgo::TOnly => write!(f, "t-only"),
            BatchingAlgo::KOnly => write!(f, "k-only"),
            BatchingAlgo::KOrT => write!(f, "k-or-t"),
            BatchingAlgo::Adaptive => write!(f, "adaptive"),
        }
    }
}

const HELP: &str = r#"
Usage: program --batching-algo <ALGO> [OPTIONS]

An inferlet for configuring the request batching strategy.

Required:
      --batching-algo <ALGO>   The batching algorithm to use. Must be one of:
                               "t-only", "k-only", "k-or-t", "adaptive".

Options:
  -t, --t <UINT>                 The timeout in milliseconds for batching.
                                 Required for "t-only" and "k-or-t".
  -k, --k <UINT>                 The batch size threshold.
                                 Required for "k-only" and "k-or-t".
  -h, --help                     Print this help information.
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // 1. Prepare the argument parser from the inferlet's environment.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 2. Handle the help flag. If present, print help and exit successfully.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // 3. Parse all required and optional arguments.
    // The `value_from_fn` uses our `FromStr` implementation on `BatchingAlgo`.
    let batching_algo: BatchingAlgo = args
        .value_from_fn("--batching-algo", BatchingAlgo::from_str)
        .map_err(|e| format!("Error: {}. Use --help for more info.", e))?;

    let t: Option<u32> = args
        .opt_value_from_str(["-t", "--t"])
        .map_err(|e| e.to_string())?;

    let k: Option<u32> = args
        .opt_value_from_str(["-k", "--k"])
        .map_err(|e| e.to_string())?;

    // 4. Validate that required parameters `t` and `k` are provided for the selected algorithm.
    match batching_algo {
        BatchingAlgo::TOnly if t.is_none() => {
            return Err("Error: --t is required for --batching-algo \"t-only\"".to_string());
        }
        BatchingAlgo::KOnly if k.is_none() => {
            return Err("Error: --k is required for --batching-algo \"k-only\"".to_string());
        }
        BatchingAlgo::KOrT if t.is_none() || k.is_none() => {
            return Err(
                "Error: --t and --k are required for --batching-algo \"k-or-t\"".to_string(),
            );
        }
        _ => {} // All other cases (Adaptive, or valid combinations) are fine.
    }

    // 5. Construct the final debug query string.
    // We default to 0 for optional values if they aren't provided, as the query format expects them.
    let k_val = k.unwrap_or(0);
    let t_val = t.unwrap_or(0);
    let query_string = format!("batching_algo:{}:{}:{}", batching_algo, k_val, t_val);

    // 6. Send the query to the inferlet runtime.
    println!("Sending debug query: \"{}\"", &query_string);
    inferlet::debug_query(&query_string).await;
    println!("Query sent successfully.");

    Ok(())
}
