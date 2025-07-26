// In src/runtime_async.rs
use crate::core; // Changed from messaging
use crate::wstd::runtime::AsyncPollable;

pub async fn receive() -> String {
    let future = core::receive(); // Changed from messaging::receive
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}

pub async fn subscribe<S: ToString>(topic: S) -> String {
    let topic = topic.to_string();
    let future = core::subscribe(&topic); // Changed from messaging::subscribe
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}
