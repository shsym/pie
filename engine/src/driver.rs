pub mod l4m;
pub mod messaging;
pub mod ping;
pub mod runtime;

use crate::instance::Id as InstanceId;
use crate::object;
use crate::object::ObjectError;
use crate::runtime::Reporter;
use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender, channel, unbounded_channel};
use tokio::task;
// Common driver routines

#[derive(Error, Debug)]
pub enum DriverError {
    #[error("Mutex lock failed")]
    LockError,

    #[error("Object error: {0}")]
    ObjectError(#[from] ObjectError),

    #[error("Event dispatcher error: {0}")]
    EventDispatcherError(#[from] anyhow::Error),

    #[error("Failed to acquire a vspace id")]
    VspacePoolAcquireFailed,

    #[error("Failed to release the vspace id")]
    VspacePoolReleaseFailed,

    #[error("Vspace not found for instance {0}")]
    VspaceNotFound(InstanceId),

    #[error("Instance already exists: {0}")]
    InstanceAlreadyExists(InstanceId),

    #[error("Instance not found: {0}")]
    InstanceNotFound(InstanceId),

    #[error("Namespace not found: {0}")]
    NamespaceNotFound(usize),

    #[error("Send error: {0}")]
    SendError(String),

    #[error("Some error: {0}")]
    Other(String),
}

pub trait Driver {
    type Command: Send + Sync + 'static;

    fn create(&mut self, inst: InstanceId) {}

    fn destroy(&mut self, inst: InstanceId) {}

    async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command);

    fn reporter(&self) -> Option<&Reporter> {
        None
    }
}

// Some common driver utilities

pub enum Operation<C> {
    Create(InstanceId),
    Destroy(InstanceId),
    Dispatch(InstanceId, C),
}

// Router -- routes commands to the appropriate driver based on the command type

pub struct AnyCommand {
    orig_type_id: TypeId,
    bx: Box<dyn Any + Send + Sync>,
}

impl AnyCommand {
    pub fn new<T: Any + Send + Sync>(cmd: T) -> Self {
        Self {
            orig_type_id: TypeId::of::<T>(),
            bx: Box::new(cmd),
        }
    }
}

pub struct Router {
    channels: HashMap<TypeId, Sender<Operation<AnyCommand>>>,
    reporter: Option<Reporter>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    pub fn install<T>(&mut self, mut driver: T)
    where
        T: Driver + 'static,
    {
        let (tx, mut rx) = channel(256);

        if self.reporter().is_none() {
            if let Some(reporter) = driver.reporter() {
                self.reporter = Some(reporter.clone());
            }
        }

        let type_id = TypeId::of::<T::Command>();

        self.channels.insert(type_id, tx);

        task::spawn(async move {
            while let Some(op) = rx.recv().await {
                match op {
                    Operation::Create(inst) => driver.create(inst),
                    Operation::Destroy(inst) => driver.destroy(inst),
                    Operation::Dispatch(inst, cmd) => {
                        driver.dispatch(inst, cmd.bx.downcast().unwrap()).await
                    }
                }
            }
        });
    }
}

impl Driver for Router {
    type Command = AnyCommand;

    fn create(&mut self, inst: InstanceId) {
        for (_, sender) in self.channels.iter() {
            sender.send(Operation::Create(inst)).unwrap();
        }
    }

    fn destroy(&mut self, inst: InstanceId) {
        for (_, sender) in self.channels.iter() {
            sender.send(Operation::Destroy(inst)).unwrap();
        }
    }

    async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
        let type_id = cmd.orig_type_id;
        self.channels
            .get_mut(&type_id)
            .ok_or(DriverError::Other("type not found".to_string()))?
            .send(Operation::Dispatch(inst, cmd))
            .await
            .unwrap();
    }

    fn reporter(&self) -> Option<&Reporter> {
        self.reporter.as_ref()
    }
}

//// ----------------- NameSelector ----------------- ////
pub struct NamedCommand<N, C>
where
    N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    name: N,
    cmd: C,
}

impl<N, C> NamedCommand<N, C>
where
    N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    pub fn new(name: N, cmd: C) -> Self {
        Self { name, cmd }
    }
}

pub struct NameSelector<T, N, C>
where
    T: Driver<Command = C>,
    N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    phantom: PhantomData<T>,
    channels: HashMap<N, Sender<Operation<C>>>,
    handles: HashMap<N, task::JoinHandle<()>>,
    reporter: Option<Reporter>,
}

impl<T, N, C> NameSelector<T, N, C>
where
    T: Driver<Command = C>,
    N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
            channels: HashMap::new(),
            handles: HashMap::new(),
            reporter: None,
        }
    }
    pub fn with(drivers: Vec<(String, T)>) -> Self {
        let mut this = Self::new();

        for (name, driver) in drivers {
            this.add_driver(&name, driver);
        }

        this
    }

    pub fn add_driver(&mut self, name: N, mut driver: T) {
        if self.reporter.is_none() {
            if let Some(reporter) = driver.reporter() {
                self.reporter = Some(reporter.clone());
            }
        }

        let (tx, mut rx) = channel(256);
        let handle = task::spawn(async move {
            while let Some(op) = rx.recv().await {
                match op {
                    Operation::Create(inst) => driver.create(inst),
                    Operation::Destroy(inst) => driver.destroy(inst),
                    Operation::Dispatch(inst, cmd) => driver.dispatch(inst, cmd.cmd).await,
                }
            }
        });

        self.channels.insert(name.to_string(), tx);
        self.handles.insert(name.to_string(), handle);
    }

    pub fn remove_driver(&mut self, name: &str) {
        self.channels.remove(name);
        if let Some(handle) = self.handles.remove(name) {
            handle.abort();
        }
    }
}

impl<T, N, C> Driver for NameSelector<T, N, C>
where
    T: Driver<Command = C>,
    N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    type Command = NamedCommand<N, C>;

    fn create(&mut self, inst: InstanceId) {
        for (_, sender) in self.channels.iter() {
            sender.send(Operation::Create(inst)).unwrap();
        }
    }

    fn destroy(&mut self, inst: InstanceId) {
        for (_, sender) in self.channels.iter() {
            sender.send(Operation::Destroy(inst)).unwrap();
        }
    }

    async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
        let NamedCommand { name, cmd } = cmd;

        if let Some(channel) = self.channels.get_mut(&name) {
            channel.send(Operation::Dispatch(inst, cmd)).await.unwrap();
        } else {
            self.reporter()
                .map(|r| r.error(inst, "selector name not found".to_string()));
        }
    }

    fn reporter(&self) -> Option<&Reporter> {
        self.reporter.as_ref()
    }
}

// TODO
pub struct LoadBalancer<T, C>
where
    T: Driver<Command = C>,
{
    drivers: Vec<T>,
    idx: usize,
}
