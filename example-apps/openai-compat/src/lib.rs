mod models;
mod handlers;
mod utils;

use pie::wstd::http::body::IncomingBody;
use pie::wstd::http::server::{Finished, Responder};
use pie::wstd::http::Request;

#[pie::server_main]
async fn main(req: Request<IncomingBody>, res: Responder) -> Finished {
    // Log connection info
    println!("Received request from {} to {}",
        req.headers().get("host").map_or("unknown", |h| h.to_str().unwrap_or("unknown")),
        req.uri().path_and_query().map_or("", |p| p.as_str()));

    match req.uri().path() {
        "/" | "/v1/completions" => handlers::handle_chat_completion(req, res).await,
        _ => handlers::handle_not_found(req, res).await,
    }
}