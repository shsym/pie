use symphony::Result;

#[symphony::main]
async fn main() -> Result<()> {
    println!("Let's chat!");

    for _ in 0..3 {
        let s = symphony::messaging_async::receive().await;
        println!("{}", s);
    }

    let responses = [
        "I'm doing well, thank you!",
        "I'm feeling great today!",
        "I'm doing fantastic!",
    ];

    for i in 0..3 {
        symphony::messaging::send(responses[i]);
    }

    println!("It was nice chatting with you. Bye!");

    Ok(())
}
