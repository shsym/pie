use crate::driver::DriverError;
use crate::instance::Id as InstanceId;
use crate::utils::IdPool;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

type SubscriptionId = usize;

#[derive(Debug)]
pub enum Command {
    /// Broadcast a message to all subscribers of a topic.
    Broadcast { topic: String, message: String },
    /// Subscribe to a topic using a sender; returns a subscription id via the oneshot.
    Subscribe {
        topic: String,
        sender: mpsc::Sender<String>,
        sub_id: oneshot::Sender<SubscriptionId>,
    },
    /// Unsubscribe from a topic using the subscription id.
    Unsubscribe {
        topic: String,
        sub_id: SubscriptionId,
    },
}

#[derive(Debug)]
pub struct Messaging {
    tx: UnboundedSender<(String, String)>,
    event_loop_handle: tokio::task::JoinHandle<()>,
    subscriptions: Arc<DashMap<String, Vec<(SubscriptionId, mpsc::Sender<String>)>>>,
    subscription_id_pool: IdPool<SubscriptionId>,
}

impl Messaging {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let subscriptions = Arc::new(DashMap::new());
        let event_loop_handle = tokio::spawn(Self::event_loop(rx, Arc::clone(&subscriptions)));

        Messaging {
            tx,
            event_loop_handle,
            subscriptions,
            subscription_id_pool: IdPool::new(SubscriptionId::MAX),
        }
    }

    pub fn channel(&self) -> UnboundedSender<(String, String)> {
        self.tx.clone()
    }

    /// Submit a command to the driver.
    ///
    pub fn submit(&mut self, inst: InstanceId, cmd: Command) -> Result<(), DriverError> {
        match cmd {
            Command::Broadcast { topic, message } => {
                // Broadcast the message. Replace unwrap with proper error handling if needed.
                self.tx
                    .send((topic, message))
                    .map_err(|e| DriverError::SendError(e.to_string()))
            }
            Command::Subscribe {
                topic,
                sender,
                sub_id,
            } => {
                // Acquire a new subscription id.
                let id = self
                    .subscription_id_pool
                    .acquire()
                    .map_err(|e| DriverError::LockError)?;

                // Insert the new subscriber into the map.
                self.subscriptions
                    .entry(topic)
                    .or_insert_with(Vec::new)
                    .push((id, sender));

                // Send back the subscription id.
                sub_id
                    .send(id)
                    .map_err(|e| DriverError::SendError(e.to_string()))
            }
            Command::Unsubscribe { topic, sub_id } => {
                if let Some(mut subscribers) = self.subscriptions.get_mut(&topic) {
                    // Remove the subscriber with the matching id.
                    subscribers.retain(|(s, _)| *s != sub_id);

                    // Remove the topic entirely if there are no more subscribers.
                    if subscribers.is_empty() {
                        self.subscriptions.remove(&topic);
                    }
                }
                // Release the subscription id back to the pool.
                self.subscription_id_pool
                    .release(sub_id)
                    .map_err(|e| DriverError::Other(e.to_string()))
            }
        }
    }

    /// The event loop that listens for broadcast messages and dispatches them to subscribers.
    async fn event_loop(
        mut rx: UnboundedReceiver<(String, String)>,
        subscriptions: Arc<DashMap<String, Vec<(SubscriptionId, mpsc::Sender<String>)>>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            if let Some(mut subscribers) = subscriptions.get_mut(&topic) {
                // Retain only the subscribers that can receive the message.
                subscribers.retain(|(_, sender)| {
                    match sender.try_send(message.clone()) {
                        Ok(_) => true,
                        Err(mpsc::error::TrySendError::Full(_)) => {
                            // The subscriber's channel is full; keep the subscription.
                            true
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            // The subscriber's channel is closed; remove the subscription.
                            false
                        }
                    }
                });

                // Remove the topic if no subscribers remain.
                if subscribers.is_empty() {
                    subscriptions.remove(&topic);
                }
            }
        }
    }
}
