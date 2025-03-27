use symphony::wstd::http::body::{BodyForthcoming, IncomingBody};
use symphony::wstd::http::server::{Finished, Responder};
use symphony::wstd::http::{IntoBody, Request, Response, StatusCode};
use symphony::wstd::io::{AsyncWrite, copy, empty};
use symphony::wstd::time::{Duration, Instant};

#[symphony::server_main]
async fn main(req: Request<IncomingBody>, res: Responder) -> Finished {
    match req.uri().path_and_query().unwrap().as_str() {
        "/wait" => wait(req, res).await,
        "/echo" => echo(req, res).await,
        "/echo-headers" => echo_headers(req, res).await,
        "/echo-trailers" => echo_trailers(req, res).await,
        "/" => home(req, res).await,
        _ => not_found(req, res).await,
    }
}

async fn home(_req: Request<IncomingBody>, res: Responder) -> Finished {
    // To send a single string as the response body, use `res::respond`.
    res.respond(Response::new("Hello, wasi:http/proxy world!\n".into_body()))
        .await
}

async fn wait(_req: Request<IncomingBody>, res: Responder) -> Finished {
    // Get the time now
    let now = Instant::now();

    // Sleep for one second.
    symphony::wstd::task::sleep(Duration::from_secs(1)).await;

    // Compute how long we slept for.
    let elapsed = Instant::now().duration_since(now).as_millis();

    // To stream data to the response body, use `res::start_response`.
    let mut body = res.start_response(Response::new(BodyForthcoming));
    let result = body
        .write_all(format!("slept for {elapsed} millis\n").as_bytes())
        .await;
    Finished::finish(body, result, None)
}

async fn echo(mut req: Request<IncomingBody>, res: Responder) -> Finished {
    // Stream data from the req body to the response body.
    let mut body = res.start_response(Response::new(BodyForthcoming));
    let result = copy(req.body_mut(), &mut body).await;
    Finished::finish(body, result, None)
}

async fn echo_headers(req: Request<IncomingBody>, responder: Responder) -> Finished {
    let mut res = Response::builder();
    *res.headers_mut().unwrap() = req.into_parts().0.headers;
    let res = res.body(empty()).unwrap();
    responder.respond(res).await
}

async fn echo_trailers(req: Request<IncomingBody>, res: Responder) -> Finished {
    let body = res.start_response(Response::new(BodyForthcoming));
    let (trailers, result) = match req.into_body().finish().await {
        Ok(trailers) => (trailers, Ok(())),
        Err(err) => (Default::default(), Err(std::io::Error::other(err))),
    };
    Finished::finish(body, result, trailers)
}

async fn not_found(_req: Request<IncomingBody>, responder: Responder) -> Finished {
    let res = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(empty())
        .unwrap();
    responder.respond(res).await
}
