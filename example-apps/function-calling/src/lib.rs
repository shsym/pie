use anyhow::{Result, anyhow};
use symphony::wstd::http::{Client, Method, Request};
use symphony::wstd::io::AsyncRead;

/// Asynchronously fetches the contents of the given URL as a String.
pub async fn fetch(url: &str) -> Result<String> {
    // Create a new HTTP client.
    let client = Client::new();

    // Build a GET request.
    let request = Request::builder()
        .uri(url)
        .method(Method::GET)
        .body(symphony::wstd::io::empty())?;

    // Send the request and get the response.
    let response = client.send(request).await?;

    // Read the response body into a buffer.
    let mut body = response.into_body();
    let mut buf = Vec::new();
    body.read_to_end(&mut buf).await?;

    // Convert the buffer into a UTF-8 string.
    String::from_utf8(buf).map_err(|e| anyhow!("Failed to convert response to UTF-8: {}", e))
}
#[symphony::main]
async fn main() -> Result<(), String> {
    // read example.com

    let url = "http://example.com";
    let response = fetch(url)
        .await
        .map_err(|e| format!("Failed to fetch {}: {}", url, e))?;

    println!("{}", response);

    Ok(())
}
