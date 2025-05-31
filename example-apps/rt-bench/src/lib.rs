
#[pie::main]
async fn main() -> Result<(), String> {

    let msg = pie::messaging_async::receive().await;

    pie::messaging::send(&msg);

    Ok(())
}
