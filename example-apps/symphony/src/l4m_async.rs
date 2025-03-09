use crate::l4m;
use crate::wstd::runtime::AsyncPollable;

// self.stream, slice::from_ref(&next_dist), 32);
pub async fn sample_top_k(stream_id: u32, dists: Vec<u32>, k: u32) -> Vec<(Vec<u32>, Vec<f32>)> {
    let res = l4m::sample_top_k(stream_id, &dists, k);
    let a = res.subscribe();
    AsyncPollable::new(a).wait_for().await;
    res.get().unwrap()
}

// pub async fn echo(msg: &str) -> String {
//     let pollable = l4m::echo(msg);
//
//     AsyncPollable::new(pollable).wait_for().await;
//     println!("done!");
//     pollable.get()
// }

// pub async fn echo2(msg: &str) -> String {
//
//     let pollable = wasi::clocks::monotonic_clock::subscribe_duration(40_000_000);
//     println!("pollable {:?}", pollable);
//     AsyncPollable::new(pollable).wait_for().await;
//     println!("done!");
//
//     msg.to_string()
// }