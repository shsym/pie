use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::time::{Duration, Instant};

pub trait BatchingStrategy: Debug + Send {
    fn update(&mut self, now: Instant);

    fn batch(&mut self, now: Instant) -> usize;
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
        KorTStrategy::new(max_wait_time, 1, Some(usize::MAX))
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
        !self.pending_items_by_stream.is_empty()
    }

    pub fn push(&mut self, stream: S, item: T, now: Instant) {
        self.pending_items_by_stream
            .entry(stream)
            .or_insert_with(VecDeque::new)
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
                    .entry(stream.clone())
                    .or_insert(curr_group);

                self.streams_by_current_group
                    .entry(curr_group)
                    .or_insert_with(Vec::new)
                    .push(stream.clone());
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
