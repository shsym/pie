#[symphony::main]
async fn main() -> Result<(), String> {
    let max_num_outputs = 32;

    // Initialize the model
    let available_models = symphony::available_models();
    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();

    // Receive role and group from the master
    let my_role = symphony::messaging_async::receive().await;
    let group = symphony::messaging_async::receive().await;

    // Define role-specific configuration
    let (system_message, section, prev_topic, next_topic) = match my_role.as_str() {
        "idea_generator" => (
            "You are a creative idea generator for stories.",
            "Idea",
            None, // No previous topic; receives initial prompt
            Some("step1"),
        ),
        "plot_developer" => (
            "You are a plot developer who creates plot outlines based on story ideas.",
            "Plot",
            Some("step1"),
            Some("step2"),
        ),
        "character_creator" => (
            "You are a character creator who invents characters for story plots.",
            "Characters",
            Some("step2"),
            Some("step3"),
        ),
        "dialogue_writer" => (
            "You are a dialogue writer who writes dialogues for stories based on plots and characters.",
            "Dialogues",
            Some("step3"),
            None, // No next topic; sends to master
        ),
        _ => return Err(format!("Unknown role: {}", my_role)),
    };

    // Construct the prompt and fetch accumulated story (if applicable)
    let (prompt, accumulated_so_far) = if my_role == "idea_generator" {
        // Initial prompt from master, e.g., "Generate a story idea about adventure."
        let initial_prompt = symphony::messaging_async::receive().await;
        (initial_prompt, None)
    } else {
        let prev_topic = prev_topic.ok_or("No previous topic defined")?;
        let accumulated =
            symphony::messaging_async::subscribe(format!("{}-{group}", prev_topic)).await;
        let instructions = match my_role.as_str() {
            "plot_developer" => "Develop a plot outline based on the idea.",
            "character_creator" => "Create characters based on the idea and plot.",
            "dialogue_writer" => "Write dialogues based on the idea, plot, and characters.",
            _ => unreachable!(), // Covered by outer match
        };
        let prompt = format!(
            "Here is the current story progress:\n{}\n\n{}",
            accumulated, instructions
        );
        (prompt, Some(accumulated))
    };

    // Create and fill the context for the model
    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill(&format!(
        "<|start_header_id|>system<|end_header_id>\n\n{}<|eot_id>",
        system_message
    ));
    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id>\n\n{}<|eot_id>",
        prompt
    ));
    ctx.fill("<|start_header_id|>assistant<|end_header_id>\n\n");

    // Generate the response
    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    // Construct the accumulated story
    let accumulated_story = match accumulated_so_far {
        Some(prev) => format!("{}\n{}: {}", prev, section, text),
        None => format!("{}: {}", section, text), // For idea_generator
    };

    // Send to the next agent or master
    if let Some(next) = next_topic {
        symphony::messaging::broadcast(&format!("{}-{group}", next), &accumulated_story);
    } else {
        symphony::messaging::send(&accumulated_story);
    }

    Ok(())
}
