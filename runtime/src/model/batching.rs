use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Batching policy that determines when to form batches
#[derive(Debug)]
pub struct ThresholdPolicy {
    max_wait_time: Duration,
    min_size: usize,
    max_size: usize,
}

impl ThresholdPolicy {
    pub fn new(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            max_wait_time,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    pub fn eager() -> Self {
        Self::new(Duration::from_secs_f32(0.0), 1, Some(usize::MAX))
    }

    pub fn t_only(max_wait_time: Duration) -> Self {
        Self::new(max_wait_time, usize::MAX, Some(usize::MAX))
    }
}

#[derive(Debug)]
pub struct BatchPolicySelector {
    policy: ThresholdPolicy,
}

impl BatchPolicySelector {
    pub fn new(policy: ThresholdPolicy) -> Self {
        Self { policy }
    }
}

#[derive(Debug)]
struct QueuedItem<T> {
    item: T,
    enqueued_at: Instant,
}

/// Generic batch scheduler that works with any item type
#[derive(Debug)]
pub struct BatchScheduler<T> {
    queue: VecDeque<QueuedItem<T>>,
    policy: BatchPolicySelector,
}

impl<T> BatchScheduler<T> {
    pub fn new(policy: BatchPolicySelector) -> Self {
        Self {
            queue: VecDeque::new(),
            policy,
        }
    }

    pub fn push(&mut self, _stream: u32, _priority: u32, item: T, now: Instant) {
        self.queue.push_back(QueuedItem {
            item,
            enqueued_at: now,
        });
    }

    pub fn schedule(&mut self, now: Instant) -> Vec<Vec<T>> {
        let mut batches = Vec::new();

        if self.queue.is_empty() {
            return batches;
        }

        let first_time = self.queue.front().unwrap().enqueued_at;
        let policy = &self.policy.policy;

        // Check if we should form a batch
        let size_ready = self.queue.len() >= policy.min_size;
        let time_ready = now.duration_since(first_time) >= policy.max_wait_time;

        if size_ready || time_ready {
            let count = self.queue.len().min(policy.max_size);
            let batch: Vec<T> = self.queue.drain(..count).map(|q| q.item).collect();
            batches.push(batch);
        }

        batches
    }

    pub fn next_poll_in(&self, now: Instant) -> Option<Duration> {
        let first = self.queue.front()?;
        let policy = &self.policy.policy;

        if self.queue.len() >= policy.min_size {
            return Some(Duration::ZERO);
        }

        let elapsed = now.duration_since(first.enqueued_at);
        if elapsed < policy.max_wait_time {
            Some(policy.max_wait_time - elapsed)
        } else {
            Some(Duration::ZERO)
        }
    }
}
