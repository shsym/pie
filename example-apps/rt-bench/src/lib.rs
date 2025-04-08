
#[symphony::main]
async fn main() -> Result<(), String> {

    let msg = symphony::messaging_async::receive().await;

    symphony::messaging::send(&msg);

    Ok(())
}
