
#[inferlet::main]
async fn main() -> Result<(), String> {

    let msg = inferlet::messaging_async::receive().await;

    inferlet::messaging::send(&msg);

    Ok(())
}
