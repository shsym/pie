use symphony::Result;
// Use Symphony's own traits if they work, as established
use symphony::wstd::io::{AsyncRead, AsyncWrite};
use symphony::wstd::iter::AsyncIterator;
use symphony::wstd::net::TcpListener;

// Added imports
use serde::Serialize; // For serializing the response structure
use serde_json;      // For converting the struct to a JSON string
use rand::{rng, Rng}; // For random generation
use rand_distr::Alphanumeric;
use std::time::{SystemTime, UNIX_EPOCH}; // For timestamp

// --- Define Response Structures (OpenAI Format) ---
#[derive(Serialize, Debug)]
struct ResponseChoiceMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct ResponseChoice {
    index: u32,
    message: ResponseChoiceMessage,
    finish_reason: String,
}

#[derive(Serialize, Debug)]
struct OpenAiResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ResponseChoice>,
    // usage: Option<UsageStats>, // We can omit usage for now
}

// --- Helper Function for Random String ---
fn generate_random_string(len: usize) -> String {
    rng()
        .sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

// --- Helper Function for Timestamp ---
fn get_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default() // Handle potential time errors simply
        .as_secs()
}


#[symphony::main]
async fn main() -> Result<()> {
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!(
        "OpenAI-like API Server listening on http://{}",
        listener.local_addr()?
    );
    println!("Send a POST request with JSON body (content ignored for now):");
    println!("  curl -X POST -H \"Content-Type: application/json\" -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}]}}' http://localhost:8080/");

    let mut incoming = listener.incoming();

    while let Some(stream_result) = incoming.next().await {
        match stream_result {
            Ok(mut stream) => {
                let peer_addr = stream.peer_addr()
                    .unwrap_or_else(|_| "unknown:0".to_string());
                println!("Accepted connection from: {}", peer_addr);

                // --- Simplified Request Reading ---
                // We read a chunk just to consume *something* from the client.
                // A real server MUST parse headers (Content-Length) and read the exact body.
                let mut buffer = [0; 1024]; // Read up to 1KB
                match stream.read(&mut buffer).await {
                    Ok(0) => {
                        println!("Client {} disconnected before sending data.", peer_addr);
                        continue; // Skip to next connection
                    }
                    Ok(bytes_read) => {
                        println!("Read {} bytes from client {} (request body ignored).", bytes_read, peer_addr);
                        // In a real app, parse `&buffer[..bytes_read]` as JSON here
                    }
                    Err(e) => {
                        eprintln!("Error reading from stream for {}: {}", peer_addr, e);
                        continue; // Skip to next connection
                    }
                };

                // --- Generate Response Content ---
                let random_content = generate_random_string(30); // Generate a 30-char random string
                let response_id = format!("chatcmpl-{}", generate_random_string(20)); // Generate a plausible ID
                let timestamp = get_unix_timestamp();

                // --- Construct the OpenAI-like Response Body ---
                let response_data = OpenAiResponse {
                    id: response_id,
                    object: "chat.completion".to_string(),
                    created: timestamp,
                    model: "symphony-mock-v1".to_string(),
                    choices: vec![
                        ResponseChoice {
                            index: 0,
                            message: ResponseChoiceMessage {
                                role: "assistant".to_string(),
                                content: random_content, // Embed the random string
                            },
                            finish_reason: "stop".to_string(),
                        }
                    ],
                };

                // --- Serialize Response to JSON ---
                let response_body_json = match serde_json::to_string(&response_data) {
                    Ok(json) => json,
                    Err(e) => {
                        eprintln!("Error serializing response JSON: {}", e);
                        // Consider sending an HTTP 500 error here
                        continue; // Skip to next connection
                    }
                };

                // --- Construct the Full HTTP Response ---
                let response = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\
                     \r\n\
                     {}", // End of headers (blank line)
                    response_body_json.len(),
                    response_body_json // The JSON payload
                );

                // --- Send the Response ---
                if let Err(e) = stream.write_all(response.as_bytes()).await {
                    eprintln!("Error writing response to {}: {}", peer_addr, e);
                } else {
                    println!("Sent JSON response to {}", peer_addr);
                }

                if let Err(e) = stream.flush().await {
                     eprintln!("Error flushing stream for {}: {}", peer_addr, e);
                }
            }
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
            }
        }
    }
    Ok(())
}