use boa_engine::{Context, Source};

#[symphony::main]
async fn main() -> Result<(), String> {
    let max_num_outputs = 32;

    // Initialize the model and context
    let available_models = symphony::available_models();
    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();
    let mut ctx = model.create_context();

    // Fill initial context with system prompt and user query
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id>\n\nYou are an assistant that can include JavaScript code in your responses to compute results. When you need to perform a computation, include the JavaScript code enclosed in ```javascript ... ``` markers in your response. The code will be executed, and the value of the last expression in the code block will be used as the result. Therefore, ensure that the last line of your code is an expression that evaluates to the desired result.\n\nDo not use `console.log()` to output the result, as it returns `undefined` and will not provide the correct value. Instead, compute the value and make it the last expression in the code block.\n\nFor example:\n\n- To compute 1 + 2:\n\n```javascript\n1 + 2\n```\n\n- For more complex computations:\n\n```javascript\nconst a = 1;\nconst b = 2;\nconst sum = a + b;\nsum\n```\n\nAvoid ending the code with `console.log()` or other statements that do not return the desired value. For instance, this is incorrect:\n\n```javascript\nconst result = 1 + 2;\nconsole.log(result);\n```\n\nThis will return `undefined`, not 3. Instead, use:\n\n```javascript\nconst result = 1 + 2;\nresult\n```\n\nWhich will return 3.<|eot_id>");    ctx.fill("<|start_header_id|>user<|end_header_id>\n\nCalculate the sum of the first 10 prime numbers.<|eot_id>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id>\n\n");

    // Generate the assistant's initial response
    let assistant_response = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    // Extract and execute JS code from the response, if present
    if let Some(js_code) = extract_js_code(&assistant_response) {
        println!("{}", js_code);
        let result = execute_js_code(&js_code);
        ctx.fill(&format!(
            "<|start_header_id|>system<|end_header_id>\n\nCode execution result: {}<|eot_id>",
            result
        ));
    } else {
        ctx.fill("<|start_header_id|>system<|end_header_id>\n\nNo code was executed.<|eot_id>");
    }

    // Generate the final assistant response incorporating the result
    ctx.fill("<|start_header_id|>assistant<|end_header_id>\n\n");
    let final_response = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    println!("{}", final_response);

    Ok(())
}

// Function to extract JS code between ```javascript ... ``` markers
fn extract_js_code(text: &str) -> Option<String> {
    let start_marker = "```javascript";
    let end_marker = "```";
    if let Some(start) = text.find(start_marker) {
        if let Some(end) = text[start + start_marker.len()..].find(end_marker) {
            let code = &text[start + start_marker.len()..start + start_marker.len() + end];
            return Some(code.trim().to_string());
        }
    }
    None
}

// Function to execute JS code and return the result or error
fn execute_js_code(code: &str) -> String {
    let mut context = Context::default();
    match context.eval(Source::from_bytes(code)) {
        Ok(result) => format!("{}", result.display()),
        Err(e) => format!("Error: {}", e),
    }
}
