use anyhow::{Result, anyhow};
use image::{DynamicImage, load_from_memory};
use symphony::Run;
use symphony::wstd::http::{Client, Method, Request};
use symphony::wstd::io::{AsyncRead, copy, empty};

/// Asynchronously fetches the contents of the given URL as a String.
pub async fn fetch_image(url: &str) -> Result<DynamicImage> {
    // Create a new HTTP client.
    let client = Client::new();

    // Build a GET request.
    let request = Request::builder()
        .uri(url)
        .method(Method::GET)
        .body(empty())?;

    // Send the request and get the response.
    let response = client.send(request).await?;

    // Read the response body into a buffer.
    let mut body = response.into_body();
    let mut buf = Vec::new();
    body.read_to_end(&mut buf).await?;

    let img = load_from_memory(&buf)?;

    // Convert the buffer into a UTF-8 string.
    Ok(img)
}
struct Multimodal;

impl Run for Multimodal {
    async fn run() -> Result<(), String> {
        // read example.com

        let url = "https://www.ilankelman.org/stopsigns/australia.jpg";
        let image = fetch_image(url)
            .await
            .map_err(|e| format!("Failed to fetch {}: {}", url, e))?;

        Ok(())
    }
}

symphony::main!(Multimodal);
