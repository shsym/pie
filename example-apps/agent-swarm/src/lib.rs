use pico_args::Arguments;
use std::ffi::OsString;

// --- Help Message ---
const HELP: &str = r#"
A single agent worker for a collaborative story-writing pipeline.
The agent's role must be specified as the first argument.

USAGE:
  agent <role> [OPTIONS]

ARGUMENTS:
  <role>    The role of the agent. Must be one of:
            - idea_generator
            - plot_developer
            - character_creator
            - dialogue_writer

OPTIONS:
  -g, --group-id <ID>         The pipeline/group ID for this agent instance. [default: 0]
  -t, --tokens-per-step <N>   Max new tokens this agent should generate. [default: 96]
  -h, --help                  Prints this help message.
"#;

// --- Agent Configuration Structure ---
struct AgentConfig {
    name: &'static str,
    system_message: &'static str,
    task_instruction: &'static str,
    section_header: &'static str,
    prev_topic: Option<&'static str>,
    next_topic: Option<&'static str>,
}

// Function to get agent configuration by role name
fn get_agent_config(role: &str) -> Result<AgentConfig, String> {
    match role {
        "idea_generator" => Ok(AgentConfig {
            name: "Story Idea Generator",
            system_message: "You are an expert idea generator on a collaborative story-writing team. Your role is to create a compelling, one-sentence story concept.",
            task_instruction: "Based on the user's request, generate a single, captivating sentence that establishes the core conflict or mystery of a story.",
            section_header: "Concept",
            prev_topic: None,
            next_topic: Some("concept_to_plot"),
        }),
        "plot_developer" => Ok(AgentConfig {
            name: "Plot Developer",
            system_message: "You are a master storyteller on a collaborative writing team. Your role is to expand a story concept into a structured plot outline.",
            task_instruction: "Read the provided story **Concept**. Your task is to write a brief plot outline with three distinct acts (Act 1: Setup, Act 2: Confrontation, Act 3: Resolution).",
            section_header: "Plot Outline",
            prev_topic: Some("concept_to_plot"),
            next_topic: Some("plot_to_chars"),
        }),
        "character_creator" => Ok(AgentConfig {
            name: "Character Creator",
            system_message: "You are an expert character designer on a collaborative writing team. Your role is to create a memorable protagonist and antagonist.",
            task_instruction: "Read the **Concept** and **Plot Outline**. Your task is to create a one-sentence description for a compelling protagonist and a formidable antagonist that fit the story.",
            section_header: "Characters",
            prev_topic: Some("plot_to_chars"),
            next_topic: Some("chars_to_dialogue"),
        }),
        "dialogue_writer" => Ok(AgentConfig {
            name: "Dialogue Writer",
            system_message: "You are a skilled dialogue writer on a collaborative writing team. Your role is to write a key piece of dialogue.",
            task_instruction: "Read all the story elements. Your task is to write a single, impactful line of dialogue spoken by the protagonist during the story's climax.",
            section_header: "Climax Dialogue",
            prev_topic: Some("chars_to_dialogue"),
            next_topic: None,
        }),
        _ => Err(format!("Unknown role: {}", role)),
    }
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- Argument Parsing ---
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

    // Parse arguments: role is required, others are optional
    let my_role: String = args
        .free_from_str()
        .map_err(|_| format!("Missing required <role> argument.\n\n{}", HELP))?;
    let group_id: u32 = args
        .opt_value_from_str(["-g", "--group-id"])
        .map_err(|e| e.to_string())?
        .unwrap_or(0);
    let tokens_per_step: usize = args
        .opt_value_from_str(["-t", "--tokens-per-step"])
        .map_err(|e| e.to_string())?
        .unwrap_or(96);

    // --- Setup ---
    let model = inferlet::get_auto_model();
    let config = get_agent_config(&my_role)?;

    // --- Determine Input and Construct Prompt ---
    let (user_prompt, accumulated_story) = if let Some(prev_topic) = config.prev_topic {
        // Subsequent agents receive the work from the previous agent
        let accumulated = inferlet::subscribe(&format!("{}-{}", prev_topic, group_id)).await;
        let prompt = format!(
            "**Previous Story Elements:**\n---\n{}\n---\n\n**Your Specific Task:**\n{}",
            accumulated, config.task_instruction
        );
        (prompt, accumulated)
    } else {
        // The first agent gets the initial prompt from the orchestrator
        let initial_prompt = inferlet::receive().await;
        let prompt = format!(
            "{}\n\nRequest: A story about {}.",
            config.task_instruction, initial_prompt
        );
        (prompt, String::new())
    };

    let mut ctx = model.create_context();
    ctx.fill_system(&config.system_message);
    ctx.fill_user(&user_prompt);


    let contribution = ctx.generate_until(tokens_per_step).await;

    // --- Format and Send Output ---
    let new_accumulated_story = format!(
        "{}\n### {}\n{}",
        accumulated_story, config.section_header, contribution
    )
    .trim()
    .to_string();

    if let Some(next_topic) = config.next_topic {
        // Send to the next agent in the pipeline
        inferlet::broadcast(
            &format!("{}-{}", next_topic, group_id),
            &new_accumulated_story,
        );
    } else {
        // Send the final completed story to the user
        inferlet::send(&new_accumulated_story);
    }

    Ok(())
}
