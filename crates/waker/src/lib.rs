mod ffi;
mod r#loom;
mod table;
mod wait;

#[cfg(not(loom))]
pub use ffi::{pie_wake, pie_wake_past};
pub use table::{
    ChannelWakers, FIRST_COMPLETION_EPOCH, MetricsSnapshot, WakeOutcome, WakerMetrics, WakerSlotId,
    WakerTable,
};
pub use wait::{Readiness, WaitFuture};

#[cfg(all(test, loom))]
mod loom_tests;
#[cfg(all(test, not(loom)))]
mod tests;
