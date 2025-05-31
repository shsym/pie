use crate::messaging;
use crate::wstd::runtime::AsyncPollable;

pub async fn receive() -> String {
    let future = messaging::receive();
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}

pub async fn subscribe<S: ToString>(topic: S) -> String {
    let topic = topic.to_string();
    let future = messaging::subscribe(&topic);
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}
