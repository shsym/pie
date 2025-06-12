use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

// Common driver routines

static SERVICE_DISPATCHER: OnceLock<Arc<Mutex<ServiceDispatcher>>> = OnceLock::new();

pub fn dispatch<C>(service_id: usize, cmd: C) -> Result<(), ServiceError>
where
    C: Any + Send + Sync + Debug + 'static,
{
    SERVICE_DISPATCHER
        .get()
        .expect("Service not initialized")
        .lock()
        .unwrap()
        .dispatch(service_id, cmd)
}

pub fn get_service_id(name: &str) -> Option<usize> {
    SERVICE_DISPATCHER
        .get()
        .expect("Service not found")
        .lock()
        .unwrap()
        .get_service_id(name)
}

pub fn add_service_runtime<T>(name: &str, driver: T) -> Result<(), ServiceError>
where
    T: Service + 'static + Send,
{
    SERVICE_DISPATCHER
        .get()
        .expect("Service not initialized")
        .lock()
        .unwrap()
        .add_service(name, driver)
}

pub fn remove_service(name: &str) -> Result<(), ServiceError> {
    SERVICE_DISPATCHER
        .get()
        .expect("Service not initialized")
        .lock()
        .unwrap()
        .remove_service(name)
}

pub fn has_service(name: &str) -> bool {
    SERVICE_DISPATCHER
        .get()
        .map(|dispatcher| {
            dispatcher
                .lock()
                .unwrap()
                .has_service(name)
        })
        .unwrap_or(false)
}

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("Driver '{0}' not found")]
    DriverNotFound(String),

    #[error("Invalid driver index: {0}")]
    InvalidDriverIndex(usize),
}
//#[async_trait]
pub trait Service: Send {
    type Command: Send + Sync + 'static;

    // fn create(&mut self, inst: InstanceId) {}
    //
    // fn destroy(&mut self, inst: InstanceId) {}

    fn handle(&mut self, cmd: Self::Command) -> impl Future<Output = ()> + Send;

    // fn reporter(&self) -> Option<&ExceptionDispatcher> {
    //     None
    // }
}

pub type AnyCommand = Box<dyn Any + Send + Sync>;

pub struct Controller {
    maps: HashMap<String, usize>,
    channels: Vec<UnboundedSender<AnyCommand>>,
}

impl Controller {
    pub fn new() -> Self {
        let builder = Self {
            maps: HashMap::new(),
            channels: Vec::new(),
        };
        //builder.add_builtin_services(listen_addr)
        builder
    }

    pub fn add<T>(mut self, name: &str, mut driver: T) -> Self
    where
        T: Service + 'static + Send,
    {
        let (tx, mut rx) = unbounded_channel();

        self.channels.push(tx);
        self.maps.insert(name.to_string(), self.channels.len() - 1);

        task::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                driver.handle(*cmd.downcast().unwrap()).await
            }
        });

        self
    }

    pub fn install(self) {
        let dispatcher = Arc::new(Mutex::new(ServiceDispatcher {
            maps: self.maps,
            channels: self.channels,
        }));
        SERVICE_DISPATCHER
            .set(dispatcher)
            .expect("Dispatcher already initialized");
    }
}

#[derive(Debug)]
pub struct ServiceDispatcher {
    maps: HashMap<String, usize>,
    channels: Vec<UnboundedSender<AnyCommand>>,
}

impl ServiceDispatcher {
    pub fn get_service_id(&self, name: &str) -> Option<usize> {
        self.maps.get(name).map(|idx| *idx)
    }

    pub fn dispatch<C>(&self, service_id: usize, cmd: C) -> Result<(), ServiceError>
    where
        C: Any + Send + Sync + Debug + 'static,
    {
        let cmd = Box::new(cmd);

        self.channels
            .get(service_id)
            .ok_or(ServiceError::InvalidDriverIndex(service_id))?
            .send(cmd)
            .unwrap();

        Ok(())
    }

    pub fn dispatch_with<C>(&self, name: &str, cmd: C) -> Result<(), ServiceError>
    where
        C: Any + Send + Sync + Debug + 'static,
    {
        let service_id = self
            .get_service_id(name)
            .ok_or(ServiceError::DriverNotFound(name.to_string()))?;

        self.dispatch(service_id, cmd)?;

        Ok(())
    }

    pub fn has_service(&self, name: &str) -> bool {
        self.maps.contains_key(name)
    }

    pub fn add_service<T>(&mut self, name: &str, mut driver: T) -> Result<(), ServiceError>
    where
        T: Service + 'static + Send,
    {
        if self.maps.contains_key(name) {
            return Err(ServiceError::DriverNotFound(format!("Service '{}' already exists", name)));
        }

        let (tx, mut rx) = unbounded_channel();
        self.channels.push(tx);
        self.maps.insert(name.to_string(), self.channels.len() - 1);

        task::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                driver.handle(*cmd.downcast().unwrap()).await
            }
        });

        Ok(())
    }

    pub fn remove_service(&mut self, name: &str) -> Result<(), ServiceError> {
        let service_id = self.maps.remove(name)
            .ok_or(ServiceError::DriverNotFound(name.to_string()))?;

        // Note: We can't actually remove from the Vec without affecting other indices
        // Instead, we'll close the channel and let the task finish naturally
        if let Some(_channel) = self.channels.get(service_id) {
            // Dropping the sender will close the channel and terminate the task
            // For now, we'll just mark it as removed from the map
            // TODO: Consider using a more sophisticated approach like Option<UnboundedSender>
        }

        Ok(())
    }
}

// Some common driver utilities
//
// pub enum Operation<C> {
//     Create(InstanceId),
//     Destroy(InstanceId),
//     Dispatch(InstanceId, C),
// }
//
// // Router -- routes commands to the appropriate driver based on the command type
//
// pub struct AnyCommand {
//     orig_type_id: TypeId,
//     bx: Box<dyn Any + Send + Sync>,
// }
//
// /// -----------
//
// impl AnyCommand {
//     pub fn new<T: Any + Send + Sync>(cmd: T) -> Self {
//         Self {
//             orig_type_id: TypeId::of::<T>(),
//             bx: Box::new(cmd),
//         }
//     }
//
//     pub fn into_inner<T: Any + Send + Sync + Debug>(self) -> Result<T, AnyCommand> {
//         if self.orig_type_id == TypeId::of::<T>() {
//             Ok(*self.bx.downcast().unwrap())
//         } else {
//             Err(self)
//         }
//     }
// }
//
// pub struct Router {
//     channels: HashMap<TypeId, Sender<Operation<AnyCommand>>>,
//     reporter: Option<ExceptionDispatcher>,
// }
//
// impl Router {
//     pub fn new() -> Self {
//         Self {
//             channels: HashMap::new(),
//         }
//     }
//
//     pub fn install<T>(&mut self, mut driver: T)
//     where
//         T: Driver + 'static,
//     {
//         let (tx, mut rx) = channel(256);
//
//         if self.reporter().is_none() {
//             if let Some(reporter) = driver.reporter() {
//                 self.reporter = Some(reporter.clone());
//             }
//         }
//
//         let type_id = TypeId::of::<T::Command>();
//
//         self.channels.insert(type_id, tx);
//
//         task::spawn(async move {
//             while let Some(op) = rx.recv().await {
//                 match op {
//                     Operation::Create(inst) => driver.create(inst),
//                     Operation::Destroy(inst) => driver.destroy(inst),
//                     Operation::Dispatch(inst, cmd) => {
//                         driver.dispatch(inst, cmd.bx.downcast().unwrap()).await
//                     }
//                 }
//             }
//         });
//     }
// }
//
// impl Driver for Router {
//     type Command = AnyCommand;
//
//     fn create(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Create(inst)).unwrap();
//         }
//     }
//
//     fn destroy(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Destroy(inst)).unwrap();
//         }
//     }
//
//     async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
//         let type_id = cmd.orig_type_id;
//         self.channels
//             .get_mut(&type_id)
//             .ok_or(DriverError::Other("type not found".to_string()))?
//             .send(Operation::Dispatch(inst, cmd))
//             .await
//             .unwrap();
//     }
//
//     fn reporter(&self) -> Option<&ExceptionDispatcher> {
//         self.reporter.as_ref()
//     }
// }
//
// //// ----------------- NameSelector ----------------- ////
// pub struct NamedCommand<N, C>
// where
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     name: N,
//     cmd: C,
// }
//
// impl<N, C> NamedCommand<N, C>
// where
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     pub fn new(name: N, cmd: C) -> Self {
//         Self { name, cmd }
//     }
// }
//
// pub struct NameSelector<T, N, C>
// where
//     T: Driver<Command = C>,
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     phantom: PhantomData<T>,
//     channels: HashMap<N, Sender<Operation<C>>>,
//     handles: HashMap<N, task::JoinHandle<()>>,
//     reporter: Option<ExceptionDispatcher>,
// }
//
// impl<T, N, C> NameSelector<T, N, C>
// where
//     T: Driver<Command = C>,
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     pub fn new() -> Self {
//         Self {
//             phantom: PhantomData,
//             channels: HashMap::new(),
//             handles: HashMap::new(),
//             reporter: None,
//         }
//     }
//     pub fn with(drivers: Vec<(String, T)>) -> Self {
//         let mut this = Self::new();
//
//         for (name, driver) in drivers {
//             this.add_driver(&name, driver);
//         }
//
//         this
//     }
//
//     pub fn add_driver(&mut self, name: N, mut driver: T) {
//         if self.reporter.is_none() {
//             if let Some(reporter) = driver.reporter() {
//                 self.reporter = Some(reporter.clone());
//             }
//         }
//
//         let (tx, mut rx) = channel(256);
//         let handle = task::spawn(async move {
//             while let Some(op) = rx.recv().await {
//                 match op {
//                     Operation::Create(inst) => driver.create(inst),
//                     Operation::Destroy(inst) => driver.destroy(inst),
//                     Operation::Dispatch(inst, cmd) => driver.dispatch(inst, cmd.cmd).await,
//                 }
//             }
//         });
//
//         self.channels.insert(name.to_string(), tx);
//         self.handles.insert(name.to_string(), handle);
//     }
//
//     pub fn remove_driver(&mut self, name: &str) {
//         self.channels.remove(name);
//         if let Some(handle) = self.handles.remove(name) {
//             handle.abort();
//         }
//     }
// }
//
// impl<T, N, C> Driver for NameSelector<T, N, C>
// where
//     T: Driver<Command = C>,
//     N: Copy + Clone + Eq + Hash + Debug + Send + Sync + 'static,
//     C: Send + Sync + 'static,
// {
//     type Command = NamedCommand<N, C>;
//
//     fn create(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Create(inst)).unwrap();
//         }
//     }
//
//     fn destroy(&mut self, inst: InstanceId) {
//         for (_, sender) in self.channels.iter() {
//             sender.send(Operation::Destroy(inst)).unwrap();
//         }
//     }
//
//     async fn dispatch(&mut self, inst: InstanceId, cmd: Self::Command) {
//         let NamedCommand { name, cmd } = cmd;
//
//         if let Some(channel) = self.channels.get_mut(&name) {
//             channel.send(Operation::Dispatch(inst, cmd)).await.unwrap();
//         } else {
//             self.reporter()
//                 .map(|r| r.error(inst, "selector name not found".to_string()));
//         }
//     }
//
//     fn reporter(&self) -> Option<&ExceptionDispatcher> {
//         self.reporter.as_ref()
//     }
// }
//
// // TODO
// pub struct LoadBalancer<T, C>
// where
//     T: Driver<Command = C>,
// {
//     drivers: Vec<T>,
//     idx: usize,
// }
