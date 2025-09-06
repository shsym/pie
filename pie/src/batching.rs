use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum BatchingConfig {
    Bounded {
        max_wait_time: Duration,
        min_size: usize,
        max_size: Option<usize>,
    },
    Triggered {
        // Will be populated by the model service at initialization.
        trigger: Option<Arc<AtomicBool>>,
        min_wait_time: Duration,
    },
}

impl BatchingConfig {
    fn to_policy(&self) -> Box<dyn BatchingPolicy> {
        match self {
            Self::Bounded {
                max_wait_time,
                min_size,
                max_size,
            } => BoundedPolicy::new(*max_wait_time, *min_size, *max_size).into_box(),
            Self::Triggered {
                trigger,
                min_wait_time,
            } => TriggeredPolicy::new(trigger.clone().unwrap(), *min_wait_time).into_box(),
        }
    }
}

/// Defines the strategy for when to form a batch from a queue of items.
trait BatchingPolicy: Debug + Send {
    /// Notifies the policy that a new item has been added.
    fn update(&mut self, now: Instant);

    /// Checks if a batch is ready according to the policy.
    /// Returns the size of the batch to be formed, or 0 if no batch is ready.
    fn poll(&mut self, now: Instant) -> usize;

    /// Hints at the optimal time to wait before the next poll.
    ///
    /// Returns `Some(duration)` to suggest sleeping for that amount of time.
    /// Returns `Some(Duration::ZERO)` if a batch is ready to be polled immediately.
    /// Returns `None` if no time-based hint is applicable (e.g., when waiting for
    /// an external trigger or if the queue is empty).
    fn next_poll_in(&self, now: Instant) -> Option<Duration>;
}

/// A policy that forms a batch only when an external trigger is fired
/// and a minimum wait time has passed since the first item arrived.
#[derive(Debug)]
pub struct TriggeredPolicy {
    count: usize,
    trigger: Arc<AtomicBool>,
    min_wait_time: Duration,
    first_item_time: Option<Instant>,
}

impl TriggeredPolicy {
    pub fn new(trigger: Arc<AtomicBool>, min_wait_time: Duration) -> Self {
        Self {
            count: 0,
            trigger,
            min_wait_time,
            first_item_time: None,
        }
    }

    fn into_box(self) -> Box<dyn BatchingPolicy> {
        Box::new(self)
    }
}

impl BatchingPolicy for TriggeredPolicy {
    fn update(&mut self, now: Instant) {
        self.count += 1;
        // If this is the first item in a new batch, record its arrival time.
        if self.count == 1 {
            self.first_item_time = Some(now);
        }
    }

    fn poll(&mut self, now: Instant) -> usize {
        if self.count == 0 {
            return 0;
        }

        // Check if the minimum wait time has passed since the first item arrived.
        let waited_long_enough = match self.first_item_time {
            Some(t0) => now.duration_since(t0) >= self.min_wait_time,
            None => self.min_wait_time == Duration::from_secs(0),
        };
        if !waited_long_enough {
            return 0;
        }

        // Atomically check and consume the trigger.
        // This ensures that even if polled multiple times while triggered,
        // a batch is formed only once per trigger event.
        if self.trigger.load(Ordering::SeqCst) {
            if self.trigger.swap(false, Ordering::SeqCst) {
                let batch_size = self.count;
                self.count = 0;
                self.first_item_time = None; // Reset for the next batch.
                return batch_size;
            }
        }
        0
    }

    fn next_poll_in(&self, now: Instant) -> Option<Duration> {
        if let Some(first_item_time) = self.first_item_time {
            let elapsed = now.duration_since(first_item_time);
            if elapsed < self.min_wait_time {
                return Some(self.min_wait_time - elapsed);
            }
        }
        // If the queue is empty, or the min_wait_time has passed, the next poll
        // depends on the external trigger, for which we cannot provide a time hint.
        None
    }
}

/// A policy that forms a batch when either a minimum number of items (`min_size`)
/// have queued up or a maximum wait time (`max_wait_time`) has passed since the
/// first item arrived. This is often called a "K-or-T" strategy.
#[derive(Debug)]
pub struct BoundedPolicy {
    max_wait_time: Duration,
    min_size: usize,
    max_size: usize,
    items: VecDeque<Instant>,
}

impl BoundedPolicy {
    pub fn new(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            max_wait_time,
            min_size,
            max_size: max_size.unwrap_or(min_size),
            items: VecDeque::new(),
        }
    }

    /// Creates a policy where batches are formed as soon as at least one item is present.
    pub fn eager() -> Self {
        BoundedPolicy::new(Duration::from_secs_f32(0.0), 1, Some(usize::MAX))
    }

    /// Creates a policy where a batch of size 1 is formed immediately for each item.
    pub fn immediate() -> Self {
        BoundedPolicy::new(Duration::from_secs_f32(0.0), 1, Some(1))
    }

    /// Creates a policy that only uses the item count threshold (`min_size`).
    /// If `max_size` is not provided, it defaults to the given `min_size`.
    pub fn k_only(min_size: usize, max_size: Option<usize>) -> Self {
        BoundedPolicy::new(Duration::MAX, min_size, max_size)
    }

    /// Creates a policy that only uses the time threshold (`max_wait_time`).
    pub fn t_only(max_wait_time: Duration) -> Self {
        BoundedPolicy::new(max_wait_time, usize::MAX, Some(usize::MAX))
    }

    /// Creates a policy that forms a batch when either the item count or time threshold is met.
    pub fn k_or_t(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        BoundedPolicy::new(max_wait_time, min_size, max_size)
    }

    fn into_box(self) -> Box<dyn BatchingPolicy> {
        Box::new(self)
    }
}

impl Clone for BoundedPolicy {
    fn clone(&self) -> Self {
        Self {
            max_wait_time: self.max_wait_time,
            min_size: self.min_size,
            max_size: self.max_size,
            items: VecDeque::new(),
        }
    }
}

impl BatchingPolicy for BoundedPolicy {
    fn update(&mut self, now: Instant) {
        self.items.push_back(now);
    }

    fn poll(&mut self, now: Instant) -> usize {
        let first = match self.items.front() {
            Some(&first) => first,
            None => return 0,
        };

        // If we haven't reached the minimum size and the wait time hasn't been exceeded, do nothing.
        if self.items.len() < self.min_size && now.duration_since(first) < self.max_wait_time {
            return 0;
        }

        // A batching condition was met. Form a batch of up to `max_size` items.
        let count = self.items.len().min(self.max_size);
        self.items.drain(..count);
        count
    }

    fn next_poll_in(&self, now: Instant) -> Option<Duration> {
        let first = match self.items.front() {
            Some(&first) => first,
            None => return None, // Queue is empty, no hint.
        };

        // If the size condition is already met, the batch is ready.
        if self.items.len() >= self.min_size {
            return Some(Duration::ZERO);
        }

        // Otherwise, the next guaranteed opportunity to form a batch is when the time limit expires.
        let elapsed = now.duration_since(first);
        if elapsed < self.max_wait_time {
            Some(self.max_wait_time - elapsed)
        } else {
            // Time has expired, the batch is ready.
            Some(Duration::ZERO)
        }
    }
}

/// A container that holds items and uses a `BatchingPolicy` to form batches.
#[derive(Debug)]
pub struct Batcher<T> {
    items: VecDeque<T>,
    policy: Box<dyn BatchingPolicy>,
}

impl<T> Batcher<T> {
    pub fn new(policy: Box<dyn BatchingPolicy>) -> Self {
        Self {
            items: VecDeque::new(),
            policy,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Pushes an item into the batcher and notifies the batching policy.
    pub fn push(&mut self, item: T, now: Instant) {
        self.items.push_back(item);
        self.policy.update(now);
    }

    /// Polls the batching policy and returns a batch if one is ready.
    pub fn poll(&mut self, now: Instant) -> Option<Vec<T>> {
        let num_items = self.policy.poll(now);
        if num_items > 0 {
            Some(self.drain_batch(num_items))
        } else {
            None
        }
    }

    /// Hints at the optimal duration to wait before the next poll.
    pub fn next_poll_in(&self, now: Instant) -> Option<Duration> {
        self.policy.next_poll_in(now)
    }

    /// Drains `count` items from the front of the queue.
    fn drain_batch(&mut self, count: usize) -> Vec<T> {
        self.items.drain(0..count).collect()
    }
}

/// Manages batching for multiple independent streams of items.
///
/// This struct enforces a strict ordering constraint: for any given stream (`S`),
/// it will only process items for one handler type (`H`) at a time. It will not
/// start processing items for a new handler type until the pending batch for the
/// current handler has been dispatched.
///
/// This is managed by a state machine using four key data structures:
/// - `queued`: A temporary holding area for items arriving from each stream.
/// - `pending`: A set of batch queues, one for each handler type (`H`), where items
///   are placed after passing the ordering check.
/// - `active_handlers`: Tracks the handler type (`H`) currently being batched for a
///   given stream (`S`). This acts as a lock to enforce ordering.
/// - `handler_to_streams`: Tracks the source stream for each individual item in a `pending`
///   batch. This is needed to release the correct stream lock in `active_handlers`
///   after a batch is dispatched.
#[derive(Debug)]
pub struct MultiStreamBatcher<H, T, S> {
    active_handlers: HashMap<S, H>,
    handler_to_streams: HashMap<H, Vec<S>>,
    queued: HashMap<S, VecDeque<(H, T, Instant)>>,
    pending: HashMap<H, Batcher<T>>,
}

impl<H, T, S> MultiStreamBatcher<H, T, S>
where
    H: Eq + Hash + Debug + Copy,
    S: Eq + Hash + Debug + Copy + Ord,
{
    pub fn new(config: HashMap<H, BatchingConfig>) -> Self {
        let pending = config
            .into_iter()
            .map(|(handler, config)| {
                let policy = config.to_policy();
                (handler, Batcher::<T>::new(policy))
            })
            .collect();

        Self {
            active_handlers: HashMap::new(),
            handler_to_streams: HashMap::new(),
            queued: HashMap::new(),
            pending,
        }
    }

    pub fn has_pending_items(&self) -> bool {
        !self.queued.is_empty() || self.pending.values().any(|q| !q.is_empty())
    }

    pub fn push(&mut self, stream: S, handler: H, item: T, now: Instant) {
        self.queued
            .entry(stream)
            .or_default()
            .push_back((handler, item, now));
    }

    /// Processes queued items and forms batches from pending queues.
    ///
    /// This method operates in two phases:
    /// 1. **Promotion**: It moves items from `queued` to the appropriate `pending`
    ///    batch queue, respecting the strict ordering constraints.
    /// 2. **Batching**: It checks each `pending` queue to see if a batch is ready
    ///    according to its scheduling policy.
    pub fn poll(&mut self, now: Instant) -> Vec<(H, Vec<T>)> {
        self.promote_queued_items();
        self.collect_ready_batches(now)
    }

    /// Hints at the optimal duration to wait before the next poll across all managed batchers.
    ///
    /// Returns the smallest wait time hint among all active batchers, or `None` if no
    /// batcher provides a time-based hint.
    pub fn next_poll_in(&self, now: Instant) -> Option<Duration> {
        self.pending
            .values()
            .filter_map(|batcher| batcher.next_poll_in(now))
            .min()
    }

    /// **Phase 1: Promote items from `queued` to `pending` queues.**
    ///
    /// This function iterates through each stream's queue, promoting consecutive items
    /// that share the same handler type. This process stops for a given stream if it
    /// encounters an item with a different handler type than the one currently "locked"
    /// for that stream in `active_handlers`.
    fn promote_queued_items(&mut self) {
        let streams = self.queued.keys().copied().collect::<Vec<_>>();

        for stream in streams {
            let queue = self.queued.get_mut(&stream).unwrap();

            // Get the handler type currently being processed for this stream, if any.
            let mut active_handler = self.active_handlers.get(&stream).copied();

            while let Some((handler, _, _)) = queue.front() {
                // If the next item's handler is different from the active one,
                // we must wait. This enforces the strict ordering.
                if let Some(active_handler) = active_handler {
                    if *handler != active_handler {
                        break;
                    }
                } else {
                    // This is the first item being processed for this stream; it sets the active handler.
                    active_handler = Some(*handler);
                    self.active_handlers.insert(stream, *handler);
                }

                // The item is eligible for promotion. Move it from `queued` to `pending`.
                let (handler, item, timestamp) = queue.pop_front().unwrap();
                self.pending
                    .get_mut(&handler)
                    .unwrap()
                    .push(item, timestamp);
                self.handler_to_streams
                    .entry(handler)
                    .or_default()
                    .push(stream);
            }
        }

        // Clean up streams whose queues have been fully drained.
        self.queued.retain(|_stream, queue| !queue.is_empty());
    }

    /// **Phase 2: Collect ready batches from `pending` queues.**
    ///
    /// Iterates through all pending queues and drains them if their batching
    /// policy indicates they are ready. When a batch is formed, it releases the
    /// stream locks in `active_handlers`.
    fn collect_ready_batches(&mut self, now: Instant) -> Vec<(H, Vec<T>)> {
        let mut ready_batches: Vec<(H, Vec<T>)> = Vec::new();

        for (handler, queue) in self.pending.iter_mut() {
            if let Some(batch) = queue.poll(now) {
                // A batch is ready. Release the handler lock for all streams
                // that contributed an item to this batch.
                let streams_in_batch = self.handler_to_streams.get_mut(handler).unwrap();
                for stream in streams_in_batch.drain(..batch.len()) {
                    self.active_handlers.remove(&stream);
                }

                ready_batches.push((*handler, batch));
            }
        }

        ready_batches
    }
}
