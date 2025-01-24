use npi::lm::{tokenize, pred, top, detokenize};
use npi::io::output;

fn main() {
    // Example input
    let input = "hello";

    // Tokenize the input string
    let tokenized = tokenize(input);

    // Generate a prediction ID
    let prediction = pred(tokenized);

    // Sample the top result
    let sampled = top(prediction);

    // Detokenize the output back to a string
    let result = detokenize(sampled);

    // Output the result
    output(&result);

    println!("Pipeline completed. Final result: {}", result);
}