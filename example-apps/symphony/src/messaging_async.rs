use crate::messaging;
use crate::wstd::runtime::AsyncPollable;

pub async fn receive() -> String {
    let future = messaging::receive();
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}
