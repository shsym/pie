use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub trait BatchingStrategy: Debug + Send {
    fn update(&mut self, now: Instant);

    fn batch(&mut self, now: Instant) -> usize;
}

pub fn eager() -> Box<dyn BatchingStrategy> {
    KorTStrategy::eager().into_box()
}

pub fn immediate() -> Box<dyn BatchingStrategy> {
    KorTStrategy::immediate().into_box()
}

pub fn k_only(min_size: usize, max_size: Option<usize>) -> Box<dyn BatchingStrategy> {
    KorTStrategy::k_only(min_size, max_size).into_box()
}

pub fn t_only(max_wait_time: Duration) -> Box<dyn BatchingStrategy> {
    KorTStrategy::t_only(max_wait_time).into_box()
}

pub fn k_or_t(
    max_wait_time: Duration,
    min_size: usize,
    max_size: Option<usize>,
) -> Box<dyn BatchingStrategy> {
    KorTStrategy::k_or_t(max_wait_time, min_size, max_size).into_box()
}

/// Creates a queue that uses an adaptive strategy to minimize latency
/// by estimating the arrival rate `lambda` online.
///
/// # Arguments
///
/// * `initial_lambda` - An initial guess for the arrival rate (items per second).
/// * `alpha` - The smoothing factor for the EWMA (e.g., 0.1). A higher value adapts faster but is less smooth.
/// * `recalc_interval` - The number of arrivals after which to recalculate the optimal batch size.
/// * `f_values` - A lookup table of `Duration`s where `f_values[n-1]` is the execution time for a batch of size `n`.
/// * `max_size` - The maximum batch size to consider. Must be `<= f_values.len()`.
pub fn adaptive(
    initial_lambda: f64,
    alpha: f64,
    recalc_interval: u32,
    f_values: Vec<Duration>,
    max_size: usize,
) -> Result<Box<dyn BatchingStrategy>, &'static str> {
    AdaptiveStrategy::new(initial_lambda, alpha, recalc_interval, f_values, max_size)
        .map(|s| s.into_box())
}

pub fn manual() -> (Box<dyn BatchingStrategy>, Arc<AtomicBool>) {
    let trigger = Arc::new(AtomicBool::new(false));
    let strategy = ManualStrategy {
        count: 0,
        trigger: trigger.clone(),
    };
    (Box::new(strategy), trigger)
}

/// A strategy that batches items only when an external trigger is fired.
#[derive(Debug)]
pub struct ManualStrategy {
    count: usize,
    trigger: Arc<AtomicBool>,
}

impl ManualStrategy {
    /// Creates a new ManualStrategy that uses the provided atomic boolean as a trigger.
    pub fn new(trigger: Arc<AtomicBool>) -> Self {
        Self { count: 0, trigger }
    }
}

impl BatchingStrategy for ManualStrategy {
    fn update(&mut self, _now: Instant) {
        //println!("item added: {:?}", self.count + 1);
        // We only need to know how many items there are, not when they arrived.
        self.count += 1;
    }

    fn batch(&mut self, _now: Instant) -> usize {
        if self.count == 0 {
            return 0;
        }
        //println!("batch fired! {:?}", self.trigger);

        // Atomically check if the trigger is set to true, and reset it to false.
        // `swap` ensures we only fire once per trigger.
        if self.trigger.swap(false, Ordering::SeqCst) {
            let batch_size = self.count;
            self.count = 0;
            batch_size
        } else {
            0
        }
    }
}

/// "K-or-T" Strategy
// 	For instance: If queue size reaches K, launch immediately; otherwise launch after T ms if K isn’t reached.
// 	This ensures that the GPU does not stay idle for too long (bounded by T) and that short bursts of arrivals form a large enough batch to get good utilization (bounded by K).
#[derive(Debug)]
pub struct KorTStrategy {
    max_wait_time: Duration,
    min_size: usize,
    max_size: usize,
    items: VecDeque<Instant>,
}

impl KorTStrategy {
    pub fn new(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            max_wait_time,
            min_size,
            max_size: max_size.unwrap_or(min_size),
            items: VecDeque::new(),
        }
    }

    pub fn into_box(self) -> Box<dyn BatchingStrategy> {
        Box::new(self)
    }

    /// Creates a queue that is "eager": batches are emitted immediately.
    pub fn eager() -> Self {
        KorTStrategy::new(Duration::from_secs_f32(0.0), 1, Some(usize::MAX))
    }

    pub fn immediate() -> Self {
        KorTStrategy::new(Duration::from_secs_f32(0.0), 1, Some(1))
    }

    /// Creates a queue that only uses the item count threshold.
    /// If `max_size` is not provided, it defaults to the given `min_size`.
    pub fn k_only(min_size: usize, max_size: Option<usize>) -> Self {
        KorTStrategy::new(Duration::MAX, min_size, max_size)
    }

    /// Creates a queue that only uses the time threshold.
    pub fn t_only(max_wait_time: Duration) -> Self {
        KorTStrategy::new(max_wait_time, usize::MAX, Some(usize::MAX))
    }

    /// Creates a queue that batches when either the time or count threshold is met.
    pub fn k_or_t(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        KorTStrategy::new(max_wait_time, min_size, max_size)
    }
}

impl Clone for KorTStrategy {
    fn clone(&self) -> Self {
        Self {
            max_wait_time: self.max_wait_time,
            min_size: self.min_size,
            max_size: self.max_size,
            items: VecDeque::new(),
        }
    }
}

impl BatchingStrategy for KorTStrategy {
    fn update(&mut self, now: Instant) {
        self.items.push_back(now);
    }

    fn batch(&mut self, now: Instant) -> usize {
        let first = match self.items.front() {
            Some(&first) => first,
            None => return 0,
        };

        // If we haven't reached the minimum size and the wait time hasn't been exceeded, do nothing.
        if self.items.len() < self.min_size && now.duration_since(first) < self.max_wait_time {
            return 0;
        }

        // Otherwise, drain up to max_size items.
        let count = self.items.len().min(self.max_size);
        self.items.drain(..count).count();
        count
    }
}

/// An adaptive strategy that determines the optimal batch size (`n*`) by modeling
/// the system as a queue to minimize average latency. It estimates the arrival
/// rate `lambda` online and periodically recalculates `n*`.
#[derive(Debug)]
pub struct AdaptiveStrategy {
    // Parameters for lambda estimation
    alpha: f64,
    avg_inter_arrival_time: Duration,
    last_arrival_time: Option<Instant>,

    // Parameters for recalculation
    f_values: Vec<Duration>,
    max_size: usize,
    updates_since_recalc: u32,
    recalc_interval: u32,

    // The calculated optimal batch size
    optimal_n: usize,
    // The queue of items
    items: VecDeque<Instant>,
}

impl AdaptiveStrategy {
    /// Creates a new `AdaptiveStrategy` by calculating an initial optimal batch size.
    pub fn new(
        initial_lambda: f64,
        alpha: f64,
        recalc_interval: u32,
        f_values: Vec<Duration>,
        max_size: usize,
    ) -> Result<Self, &'static str> {
        if max_size == 0 || max_size > f_values.len() {
            return Err("max_size must be > 0 and <= f_values.len()");
        }

        // Calculate the initial optimal_n based on the provided guess
        let initial_n = Self::calculate_optimal_n(initial_lambda, &f_values, max_size)
            .ok_or("Could not find a stable initial batch size.")?;

        Ok(Self {
            alpha,
            // Initialize avg inter-arrival time based on the initial lambda guess
            avg_inter_arrival_time: Duration::from_secs_f64(1.0 / initial_lambda),
            last_arrival_time: None,
            f_values,
            max_size,
            updates_since_recalc: 0,
            recalc_interval,
            optimal_n: initial_n,
            items: VecDeque::new(),
        })
    }

    /// Performs the core calculation to find the optimal batch size `n*`.
    fn calculate_optimal_n(lambda: f64, f_values: &[Duration], max_size: usize) -> Option<usize> {
        let mut best_n = 0;
        let mut min_latency = f64::MAX;

        for n in 1..=max_size {
            let n_f64 = n as f64;
            // Look up execution time from the vector
            let fn_duration = f_values[n - 1];
            let fn_seconds = fn_duration.as_secs_f64();

            if lambda * fn_seconds >= n_f64 {
                continue;
            }

            let w_batch = (n_f64 - 1.0) / (2.0 * lambda);
            let w_queue = n_f64 / (2.0 * lambda * (n_f64 - lambda * fn_seconds));
            let w_proc = fn_seconds;
            let current_latency = w_batch + w_queue + w_proc;

            if current_latency < min_latency {
                min_latency = current_latency;
                best_n = n;
            }
        }

        if best_n > 0 { Some(best_n) } else { None }
    }

    /// Re-evaluates `optimal_n` using the latest `lambda` estimate.
    fn recalculate(&mut self) {
        let avg_iat_secs = self.avg_inter_arrival_time.as_secs_f64();
        if avg_iat_secs < 1e-9 {
            return;
        } // Avoid division by zero
        let lambda_estimate = 1.0 / avg_iat_secs;

        // If a new stable n is found, update it. Otherwise, keep the old one.
        if let Some(new_n) =
            Self::calculate_optimal_n(lambda_estimate, &self.f_values, self.max_size)
        {
            self.optimal_n = new_n;
        }
    }

    pub fn into_box(self) -> Box<dyn BatchingStrategy> {
        Box::new(self)
    }
}

impl BatchingStrategy for AdaptiveStrategy {
    fn update(&mut self, now: Instant) {
        self.items.push_back(now);

        // Update the EWMA of the inter-arrival time
        if let Some(last_time) = self.last_arrival_time {
            let current_iat = now.duration_since(last_time);
            let old_avg_secs = self.avg_inter_arrival_time.as_secs_f64();
            let new_avg_secs =
                self.alpha * current_iat.as_secs_f64() + (1.0 - self.alpha) * old_avg_secs;
            self.avg_inter_arrival_time = Duration::from_secs_f64(new_avg_secs);
        }
        self.last_arrival_time = Some(now);

        // Check if it's time to recalculate the optimal batch size
        self.updates_since_recalc += 1;
        if self.updates_since_recalc >= self.recalc_interval {
            self.recalculate();
            self.updates_since_recalc = 0;
        }
    }

    fn batch(&mut self, _now: Instant) -> usize {
        // Fire a batch of size `optimal_n` as soon as enough items are available.
        if self.items.len() >= self.optimal_n {
            self.items.drain(..self.optimal_n);
            self.optimal_n
        } else {
            0
        }
    }
}

#[derive(Debug)]
pub struct BatchQueue<T> {
    // cmd, timestamp, response_sender
    items: VecDeque<T>,

    strategy: Box<dyn BatchingStrategy>,
}

impl<T> BatchQueue<T> {
    pub fn new(strategy: Box<dyn BatchingStrategy>) -> Self {
        Self {
            items: VecDeque::new(),
            strategy,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Push an item with the current timestamp.
    pub fn push(&mut self, item: T, now: Instant) {
        self.items.push_back(item);
        self.strategy.update(now);
    }

    pub fn batch(&mut self, now: Instant) -> Option<Vec<T>> {
        let num_items = self.strategy.batch(now);
        if num_items > 0 {
            Some(self.drain_batch(num_items))
        } else {
            None
        }
    }

    /// Drains up to `max_size` items from the front of the queue.
    fn drain_batch(&mut self, count: usize) -> Vec<T> {
        self.items.drain(0..count).collect()
    }
}

pub trait Batchable<G> {
    fn strategy(&self) -> Box<dyn BatchingStrategy>;

    fn group(&self) -> G;
}

#[derive(Debug)]
pub struct Batcher<T, S, G> {
    current_group_by_stream: HashMap<S, G>,
    streams_by_current_group: HashMap<G, Vec<S>>,
    pending_items_by_stream: HashMap<S, VecDeque<(T, Instant)>>,
    batch_queues_by_group: HashMap<G, BatchQueue<T>>,
}

impl<T, S, G> Batcher<T, S, G>
where
    T: Batchable<G>,
    S: Eq + Hash + Debug + Copy + Ord,
    G: Eq + Hash + Debug + Copy,
{
    pub fn new() -> Self {
        Self {
            current_group_by_stream: HashMap::new(),
            streams_by_current_group: HashMap::new(),
            pending_items_by_stream: HashMap::new(),
            batch_queues_by_group: HashMap::new(),
        }
    }

    pub fn has_pending_items(&self) -> bool {
        // First, check the initial holding pen.
        if !self.pending_items_by_stream.is_empty() {
            return true;
        }

        // IMPORTANT: Also check if any of the staged queues have items.
        self.batch_queues_by_group.values().any(|q| !q.is_empty())
    }

    pub fn push(&mut self, stream: S, item: T, now: Instant) {
        self.pending_items_by_stream
            .entry(stream)
            .or_default()
            .push_back((item, now));
    }

    pub fn batch(&mut self, now: Instant) -> Vec<(G, Vec<T>)> {
        // Horizontal batching: group commands by stream and type.
        let mut empty_streams = Vec::new();

        // Sort by stream priority
        let mut streams_sorted: Vec<S> = self.pending_items_by_stream.keys().copied().collect();
        streams_sorted.sort();

        for stream in streams_sorted {
            let queue = self.pending_items_by_stream.get_mut(&stream).unwrap();

            // non-flushed commands sharing the same stream in the cmd_batcher
            // None -> no commands in the batch queue with the same stream
            let mut prev_group = self.current_group_by_stream.get(&stream).cloned();

            while !queue.is_empty() {
                let curr_group = queue.front().unwrap().0.group();

                // Vertical batching: Same kind of consecutive commands are batched together.
                // if the current command is different from the previous one, stop batching.
                if let Some(prev_group) = prev_group {
                    if curr_group != prev_group {
                        break;
                    }
                }
                prev_group = Some(curr_group);

                let (item, timestamp) = queue.pop_front().unwrap();
                self.batch_queues_by_group
                    .entry(curr_group)
                    .or_insert(BatchQueue::<T>::new(item.strategy()))
                    .push(item, timestamp);

                self.current_group_by_stream
                    .entry(stream)
                    .or_insert(curr_group);

                self.streams_by_current_group
                    .entry(curr_group)
                    .or_default()
                    .push(stream);
            }

            if queue.is_empty() {
                empty_streams.push(stream);
            }
        }

        for stream in empty_streams {
            self.pending_items_by_stream.remove(&stream);
        }

        // Batch commands and return them.
        let mut batches = Vec::new();

        for (grp, queue) in self.batch_queues_by_group.iter_mut() {
            if let Some(cmds) = queue.batch(now) {
                for stream in self
                    .streams_by_current_group
                    .get_mut(grp)
                    .unwrap()
                    .drain(..cmds.len())
                {
                    self.current_group_by_stream.remove(&stream);
                }

                batches.push((*grp, cmds));
            }
        }

        batches
    }
}
