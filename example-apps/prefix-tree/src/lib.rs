use futures::future::join_all;
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;
// It creates a 1 * 2 * 2 * 2 = 8 prompt tree.
//
// The structure is as follows:
//
//                  [System Prompt]
//                 /                 \
//        [Photosynthesis]      [Cellular Respiration]
//        /              \          /                  \
//   [ELI5]         [High School] [Location in Cell]  [Main Products]
//   /    \           /      \      /         \         /       \
// [Chef] [Sunlight] [Equation] [Algae] [Mitochondria] [P&A] [ATP] [CO2]
//
// Each of the 8 leaf nodes is then used to generate a short text concurrently.

const HELP: &str = "\
Usage: prefix_tree [OPTIONS]

A program to test prefix tree caching by generating text from 8 related prompts concurrently.

Options:
  -n, --num-tokens <TOKENS>  Sets the number of tokens to generate for each prompt [default: 32]
  -h, --help                 Prints this help message
";

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- Argument Parsing ---
    // Parse command-line arguments to make the number of output tokens configurable.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // If the help flag is present, print the help message and exit.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // Parse the optional '--num-tokens' argument. Defaults to 32 if not provided.
    let max_num_outputs_per_prompt: usize = args
        .opt_value_from_str(["-n", "--num-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(32);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let mut ctx_root = model.create_context();

    // 1. --- Root Context (Level 0) ---
    // All forks will share this initial system prompt.
    ctx_root.fill("<|begin_of_text|>");
    ctx_root.fill(
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, friendly, and knowledgeable science tutor for students of all ages. Your goal is to explain complex biological concepts in a clear, accessible, and engaging manner, tailoring your language to the specified audience.<|eot_id|>",
    );

    // 2. --- First Level Forks (Level 1) ---
    // We create two main branches from the root context.
    let mut ctx_photo = ctx_root.fork();
    ctx_photo.fill(
        "<|start_header_id|>user<|end_header_id|>\n\nI'm curious about the fundamental process of photosynthesis. Could you provide a detailed overview of how plants create their own food using sunlight, water, and carbon dioxide?<|eot_id|>",
    );

    let mut ctx_resp = ctx_root.fork();
    ctx_resp.fill(
        "<|start_header_id|>user<|end_header_id|>\n\nNow, could you explain the equally important process of cellular respiration? I'd like to understand how organisms, including plants and animals, break down glucose to release the energy needed for life.<|eot_id|>",
    );

    // 3. --- Second Level Forks (Level 2) ---
    // Fork each of the two branches into two more, creating 4 distinct contexts.
    let mut ctx_photo_eli5 = ctx_photo.fork();
    ctx_photo_eli5
        .fill("<|start_header_id|>user<|end_header_id|>\n\nThat sounds complicated. Could you simplify it significantly for me? Please explain the core idea in a way that a curious 5-year-old child could easily grasp and remember. Use a simple analogy.<|eot_id|>");

    let mut ctx_photo_hs = ctx_photo.fork();
    ctx_photo_hs.fill("<|start_header_id|>user<|end_header_id|>\n\nThank you. Now, could you provide a more technical explanation suitable for a high school biology student? I'm familiar with basic cell biology and chemistry, so please include relevant terminology like chloroplasts, chlorophyll, and light-dependent reactions.<|eot_id|>");

    let mut ctx_resp_loc = ctx_resp.fork();
    ctx_resp_loc.fill(
        "<|start_header_id|>user<|end_header_id|>\n\nI'm interested in the specific location within the cell where this process occurs. Can you describe the organelles involved and why their specific structures are uniquely suited for this essential energy-releasing function?<|eot_id|>",
    );

    let mut ctx_resp_prod = ctx_resp.fork();
    ctx_resp_prod
        .fill("<|start_header_id|>user<|end_header_id|>\n\nFocusing on the outputs of this metabolic reaction, what are the primary products that result from this process? Please list and briefly describe the significance of each of these molecules for the cell.<|eot_id|>");

    // 4. --- Third Level Forks (Level 3) ---
    // Fork each of the four contexts into two final leaf nodes, resulting in 8 total prompts.
    // These are the contexts we will generate from.
    let mut prompts = vec![];

    // Photosynthesis -> ELI5 -> ...
    let mut p1 = ctx_photo_eli5.fork();
    p1.fill("<|start_header_id|>user<|end_header_id|>\n\nTo make it really fun, please begin your explanation with the exact phrase 'Plants are like little chefs...' and continue that cooking analogy to describe how they make their sugary food.<|eot_id|>");
    prompts.push(p1);

    let mut p2 = ctx_photo_eli5.fork();
    p2.fill("<|start_header_id|>user<|end_header_id|>\n\nLet's zoom in on the energy source for this recipe. Can you specifically detail the crucial role that sunlight plays in this process? Explain what the sun's energy does and why it's so important for the plant's 'kitchen'.<|eot_id|>");
    prompts.push(p2);

    // Photosynthesis -> High School -> ...
    let mut p3 = ctx_photo_hs.fork();
    p3.fill("<|start_header_id|>user<|end_header_id|>\n\nFor a more precise, scientific understanding, please provide the balanced chemical equation for the overall photosynthetic reaction. Also, briefly explain what each part of the equation represents in the context of the plant's metabolism.<|eot_id|>");
    prompts.push(p3);

    let mut p4 = ctx_photo_hs.fork();
    p4.fill(
        "<|start_header_id|>user<|end_header_id|>\n\nHow does this process in terrestrial plants compare to what happens in aquatic organisms like algae or cyanobacteria? Are there any significant differences in the mechanism, pigments used, or the cellular location?<|eot_id|>",
    );
    prompts.push(p4);

    // Cellular Respiration -> Location -> ...
    let mut p5 = ctx_resp_loc.fork();
    p5.fill(
        "<|start_header_id|>user<|end_header_id|>\n\nPlease elaborate specifically on the role of the mitochondria. Describe its inner and outer membranes and the matrix, and explain how this structure makes it the perfect 'powerhouse' of the cell during this process.<|eot_id|>",
    );
    prompts.push(p5);

    let mut p6 = ctx_resp_loc.fork();
    p6.fill("<|start_header_id|>user<|end_header_id|>\n\nIs this metabolic pathway entirely identical in both plant and animal cells? Please compare and contrast the process, highlighting any key similarities or differences in where or how cellular respiration occurs in these two major kingdoms.<|eot_id|>");
    prompts.push(p6);

    // Cellular Respiration -> Products -> ...
    let mut p7 = ctx_resp_prod.fork();
    p7.fill("<|start_header_id|>user<|end_header_id|>\n\nOne of the key products is usable energy. Could you explain in detail the role of adenosine triphosphate (ATP) as the main energy currency? How is it synthesized and then used by the cell to power its activities?<|eot_id|>");
    prompts.push(p7);

    let mut p8 = ctx_resp_prod.fork();
    p8.fill(
        "<|start_header_id|>user<|end_header_id|>\n\nI understand that carbon dioxide is considered a waste product of this process. Can you elaborate on what exactly happens to this CO2? How does the organism expel it, and what is its ultimate fate in the larger ecosystem?<|eot_id|>",
    );
    prompts.push(p8);

    // 5. --- Prepare and Execute Futures Concurrently ---
    // Add the final assistant prompt and create a future for each of the 8 leaf contexts.
    println!(
        "--- Starting concurrent generation for 8 prompts ({} tokens each) ---",
        max_num_outputs_per_prompt
    );
    let generation_futures: Vec<_> = prompts
        .into_iter()
        .map(|mut ctx| async move {
            ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");
            // The async move block takes ownership of `ctx`, resolving the lifetime issue.
            // We then await the future returned by `generate_until`.
            ctx.generate_until(max_num_outputs_per_prompt)
                .await
        })
        .collect();

    // Use join_all to run all generation tasks in parallel.
    let results = join_all(generation_futures).await;

    println!(
        "\n--- All 8 generations completed in {:?} ---\n",
        start.elapsed()
    );

    // 6. --- Print Results ---
    for (i, output_text) in results.iter().enumerate() {
        println!("Prompt #{}:\n{:?}\n", i + 1, output_text);
    }

    Ok(())
}
