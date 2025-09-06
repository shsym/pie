use inferlet::Context;
use inferlet::brle::Brle;
use inferlet::traits::Forward;
use inferlet::traits::tokenize::Tokenizer;
use std::time::Instant;

/// Implements hierarchical attention prefill based on XML structure.
///
/// This function parses an XML string to understand its tree structure. It then
/// constructs a custom attention mask for a single, large forward pass. The mask
/// adheres to the following hierarchical rules:
///
/// 1.  **Intra-Node Attention**: Tokens within the same XML node can attend to each other (causally).
/// 2.  **Parent-to-Child Attention**: Tokens in a parent node can attend to tokens in their direct child nodes.
/// 3.  **Masked Relations**: All other relationships are masked.
///
/// # Arguments
/// * `ctx`: A mutable reference to the context to be prefilled.
/// * `xml_text`: A string slice containing the XML document to process.
pub async fn prefill_with_hierarchical_attention(ctx: &mut Context, xml_text: &str) {
    #[derive(Debug, Clone)]
    struct XmlNode {
        id: usize,
        parent_id: Option<usize>,
        token_start: usize,
        token_end: usize,
    }

    struct ParsedXml {
        tokens: Vec<u32>,
        nodes: Vec<XmlNode>,
        token_to_node_id: Vec<usize>,
    }

    /// A simple, stack-based XML parser to build the document tree.
    fn parse_xml(xml_text: &str, tokenizer: &Tokenizer) -> ParsedXml {
        let mut tokens = Vec::new();
        let mut nodes = Vec::new();
        let mut token_to_node_id = Vec::new();
        let mut node_stack: Vec<usize> = Vec::new();
        let mut next_node_id = 0;

        // Create the root node for the whole document
        let root_node = XmlNode {
            id: next_node_id,
            parent_id: None,
            token_start: 0,
            token_end: 0,
        };
        nodes.push(root_node);
        node_stack.push(next_node_id);
        next_node_id += 1;

        let mut last_pos = 0;
        for (i, _) in xml_text.match_indices('<') {
            if i > last_pos {
                let text = &xml_text[last_pos..i];
                let text_tokens = tokenizer.tokenize(text);
                let node_id = *node_stack.last().unwrap();
                token_to_node_id.resize(tokens.len() + text_tokens.len(), node_id);
                tokens.extend(text_tokens);
            }

            let end = xml_text[i..].find('>').unwrap() + i;
            let tag_content = &xml_text[i + 1..end];
            let tag_tokens = tokenizer.tokenize(&xml_text[i..=end]);
            let current_node_id = *node_stack.last().unwrap();
            token_to_node_id.resize(tokens.len() + tag_tokens.len(), current_node_id);
            tokens.extend(tag_tokens);

            if tag_content.starts_with('/') {
                let popped_node_id = node_stack.pop().unwrap();
                nodes[popped_node_id].token_end = tokens.len();
            } else if !tag_content.ends_with('/') {
                let parent_id = *node_stack.last().unwrap();
                let new_node = XmlNode {
                    id: next_node_id,
                    parent_id: Some(parent_id),
                    token_start: tokens.len(),
                    token_end: 0,
                };
                nodes.push(new_node);
                node_stack.push(next_node_id);
                next_node_id += 1;
            }
            last_pos = end + 1;
        }
        nodes[0].token_end = tokens.len();

        ParsedXml {
            tokens,
            nodes,
            token_to_node_id,
        }
    }

    // 1. Parse the XML to get the structural mapping.
    let parsed = parse_xml(xml_text, &ctx.tokenizer);
    let num_tokens = parsed.tokens.len();
    if num_tokens == 0 {
        return;
    }

    // 2. Build the hierarchical attention mask.
    let mut attention_masks_rle: Vec<Vec<u32>> = Vec::with_capacity(num_tokens);
    for j in 0..num_tokens {
        let mut masked: Vec<bool> = vec![true; j + 1]; // all mask
        let node_j_id = parsed.token_to_node_id[j];
        let node_j = &parsed.nodes[node_j_id];

        for i in 0..=j {
            // Causal constraint
            let node_i_id = parsed.token_to_node_id[i];
            let node_i = &parsed.nodes[node_i_id];

            if node_i.id == node_j.id || node_i.parent_id == Some(node_j.id) {
                masked[i] = false; // no mask
            }
        }
        let b = Brle::from_slice(&masked);
        attention_masks_rle.push(b.buffer);
    }

    // 3. Prepare for and execute the forward pass.
    let mut pending_token_ids = parsed.tokens;
    let position_ids: Vec<u32> = (0..num_tokens as u32).collect();
    ctx.grow_kv_pages(num_tokens);

    let p = ctx.queue.create_forward_pass();

    p.kv_cache(&ctx.kv_pages, ctx.kv_page_last_len);
    p.attention_mask(&attention_masks_rle);
    p.input_tokens(&pending_token_ids, &position_ids);

    let _ = p.execute().await;

    let next_token_id = pending_token_ids.pop().unwrap();
    // 4. Update the context's state.
    ctx.token_ids.extend(&pending_token_ids);
    ctx.position_ids.extend(&position_ids);
    ctx.fill_token(next_token_id);
}

#[inferlet::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();
    let model = inferlet::get_auto_model();
    let mut ctx = Context::new(&model);

    let xml_document = r#"<article>
  <title>The Future of AI</title>
  <author>Jane Doe</author>
  <abstract>
    <paragraph>This paper explores the potential impact of artificial intelligence on society.</paragraph>
    <paragraph>We discuss advancements in machine learning and natural language processing.</paragraph>
  </abstract>
  <section>
    <heading>Introduction</heading>
    <content>AI is a rapidly evolving field with profound implications.</content>
  </section>
</article>"#;

    println!("--- Input XML ---\n{}\n-----------------", xml_document);

    // Call the standalone prefill function
    prefill_with_hierarchical_attention(&mut ctx, xml_document).await;

    println!("Hierarchical prefill complete. Starting generation...");

    let elapsed = start.elapsed();

    println!("Total elapsed: {:?}", elapsed,);

    Ok(())
}
