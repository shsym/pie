use pico_args::Arguments;
use std::ffi::OsString;

const HELP: &str = r#"
Usage: program [OPTIONS]

An inferlet for benchmarking the execution latency.

Options:
  -m, --message <STRING>   The message to send.
                           (default: "Hello from the inferlet!")
  -l, --layer <STRING>     The layer to send the message to. (either 'control' or 'inference')
                            (default: "control")
  -h, --help               Print help information.
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

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let msg = args
        .opt_value_from_str(["-m", "--message"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Hello from the inferlet!".to_string());

    let layer = args
        .opt_value_from_str(["-l", "--layer"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Hello from the inferlet!".to_string());


    let res = inferlet::debug_query("get_cache_dir").await;
    println!("{:?}", res);

    let model = inferlet::get_auto_model();

    let res = model.create_queue().debug_query("ping").await;


    println!("{:?}", res);

    inferlet::send(&msg);

    Ok(())
}
