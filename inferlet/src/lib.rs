pub use crate::bindings::pie::inferlet::core::Priority;
use crate::bindings::pie::inferlet::{adapter, core, evolve, forward, image, tokenize};
pub use crate::bindings_app::{export, exports::pie::inferlet::run::Guest as RunSync};
pub use crate::context::Context;
pub use crate::chat::ChatFormatter;
use crate::wstd::runtime::AsyncPollable;
pub use anyhow::Result;
pub use inferlet_macros::main;
pub use inferlet_macros::server_main;
use std::collections::HashSet;
use std::rc::Rc;
pub use wasi;
use wasi::exports::http::incoming_handler::{IncomingRequest, ResponseOutparam};
pub use wstd;

pub mod brle;
pub mod context;
pub mod chat;
pub mod drafter;
mod pool;
pub mod sampler;
pub mod stop_condition;
pub mod traits;

pub mod bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "imports",
        pub_export_macro: true,
        with: {
             "wasi:io/poll@0.2.4": wasi::io::poll,
        },
        generate_all,
    });
}

pub mod bindings_app {
    wit_bindgen::generate!({
        path: "wit",
        world: "inferlet",
        pub_export_macro: true,
        default_bindings_module: "pie::bindings",
        with: {
             "wasi:io/poll@0.2.4": wasi::io::poll,
        },
    });
}

pub mod bindings_server {
    wit_bindgen::generate!({
        path: "wit",
        world: "inferlet-server",
        pub_export_macro: true,
        default_bindings_module: "pie::bindings",
        with: {
            "wasi:io/poll@0.2.4": wasi::io::poll,
            "wasi:clocks/monotonic-clock@0.2.4": wasi::clocks::monotonic_clock,
            "wasi:io/error@0.2.4": wasi::io::error,
            "wasi:io/streams@0.2.4": wasi::io::streams,
            "wasi:http/types@0.2.4": wasi::http::types,
        },
        generate_all,
    });
}

#[derive(Clone, Debug)]
pub struct Queue {
    pub(crate) inner: Rc<core::Queue>,
    service_id: u32,
}

/// Represents a specific model instance, providing access to its metadata and functionality.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) inner: Rc<core::Model>,
}

pub enum Resource {
    KvPage = 0,
    Embed = 1,
    Adapter = 2,
}

/// Returns the runtime version string.
pub fn get_version() -> String {
    core::get_version()
}

/// Returns a unique identifier for the running instance.
pub fn get_instance_id() -> String {
    core::get_instance_id()
}

/// Retrieves POSIX-style CLI arguments passed to the inferlet from the remote user client.
pub fn get_arguments() -> Vec<String> {
    core::get_arguments()
}

/// Retrieve a model by its name.
///
/// Returns `None` if no model with the specified name is found.
pub fn get_model(name: &str) -> Option<Model> {
    core::get_model(name).map(|inner| Model {
        inner: Rc::new(inner),
    })
}

pub fn get_auto_model() -> Model {
    let models = get_all_models();
    if models.is_empty() {
        panic!("No models available");
    }

    // choose the first model
    let model_name = models[0].clone();
    let model = get_model(&model_name).unwrap();
    model
}

/// Get a list of all available model names.
pub fn get_all_models() -> Vec<String> {
    core::get_all_models()
}

/// Get names of models that have all specified traits (e.g. "input_text", "tokenize").
pub fn get_all_models_with_traits(traits: &[String]) -> Vec<String> {
    core::get_all_models_with_traits(traits)
}

/// Sends a message to the remote user client.
pub fn send(message: &str) {
    core::send(message)
}

/// Receives an incoming message from the remote user client.
///
/// This is an asynchronous operation that returns a `ReceiveResult`.
pub async fn receive() -> String {
    let future = core::receive(); // Changed from messaging::receive
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}

/// Publishes a message to a topic, broadcasting it to all subscribers.
pub fn broadcast(topic: &str, message: &str) {
    core::broadcast(topic, message)
}

/// Subscribes to a topic and returns a `Subscription` handle.
pub async fn subscribe<S: ToString>(topic: S) -> String {
    let topic = topic.to_string();
    let future = core::subscribe(&topic); // Changed from messaging::subscribe
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}

/// Retrieves a value from the persistent store for a given key.
///
/// Returns `Some(value)` if the key exists, or `None` if it does not.
pub fn store_get(key: &str) -> Option<String> {
    core::store_get(key)
}

/// Sets a value in the persistent store for a given key.
///
/// This will create a new entry or overwrite an existing one.
pub fn store_set(key: &str, value: &str) {
    core::store_set(key, value)
}

/// Deletes a key-value pair from the store.
///
/// If the key does not exist, this function does nothing.
pub fn store_delete(key: &str) {
    core::store_delete(key)
}

/// Checks if a key exists in the store.
///
/// Returns `true` if the key exists, and `false` otherwise.
pub fn store_exists(key: &str) -> bool {
    core::store_exists(key)
}

/// Returns a list of all keys currently in the store.
pub fn store_list_keys() -> Vec<String> {
    core::store_list_keys()
}

/// Executes a debug command and returns the result as a string.
pub async fn debug_query(query: &str) -> String {
    let future = core::debug_query(query);
    let pollable = future.pollable();
    AsyncPollable::new(pollable).wait_for().await;
    future.get().unwrap()
}

impl Model {
    /// Returns the model's name (e.g. "llama-3.1-8b-instruct").
    pub fn get_name(&self) -> String {
        self.inner.get_name()
    }

    /// Returns the full set of model traits.
    pub fn get_traits(&self) -> Vec<String> {
        self.inner.get_traits()
    }

    pub fn has_traits(&self, required_traits: &[&str]) -> bool {
        let available_traits_vec = self.get_traits();

        let available_traits: HashSet<&str> =
            available_traits_vec.iter().map(String::as_str).collect();

        // Find any required traits that are not in the available set.
        let missing: Vec<_> = required_traits
            .iter()
            .filter(|t| !available_traits.contains(*t))
            .cloned()
            .collect();

        missing.is_empty()
    }

    /// Returns a human-readable description of the model.
    pub fn get_description(&self) -> String {
        self.inner.get_description()
    }

    /// Returns the prompt formatting template in Tera format.
    pub fn get_prompt_template(&self) -> String {
        self.inner.get_prompt_template()
    }

    pub fn get_stop_tokens(&self) -> Vec<String> {
        self.inner.get_stop_tokens()
    }

    /// Gets the service ID for the model.
    pub fn get_service_id(&self) -> u32 {
        self.inner.get_service_id()
    }

    pub fn get_kv_page_size(&self) -> u32 {
        self.inner.get_kv_page_size()
    }

    /// Create a new command queue for this model.
    pub fn create_queue(&self) -> Queue {
        Queue {
            inner: Rc::new(self.inner.create_queue()),
            service_id: self.inner.get_service_id(),
        }
    }

    pub fn create_context(&self) -> Context {
        Context::new(self)
    }
}

impl Queue {
    /// Gets the service ID for the queue.
    pub fn get_service_id(&self) -> u32 {
        self.service_id
    }

    /// Begins a synchronization process for the queue, returning a `SynchronizationResult`.
    pub async fn synchronize(&self) -> bool {
        let future = self.inner.synchronize(); // Changed from messaging::receive
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
    }

    /// Change the queue's priority.
    pub fn set_priority(&self, priority: Priority) {
        self.inner.set_priority(priority)
    }

    pub async fn debug_query(&self, query: &str) -> String {
        let future = self.inner.debug_query(query);
        let pollable = future.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future.get().unwrap()
    }

    pub fn allocate_resources(&self, resource: Resource, count: u32) -> Vec<u32> {
        core::allocate_resources(&self.inner, resource as u32, count)
    }

    pub fn deallocate_resources(&self, resource: Resource, ptrs: &[u32]) {
        core::deallocate_resources(&self.inner, resource as u32, ptrs)
    }

    pub fn export_resource(&self, resource: Resource, ptrs: &[u32], name: &str) {
        core::export_resources(&self.inner, resource as u32, ptrs, name)
    }

    pub fn import_resource(&self, resource: Resource, name: &str) -> Vec<u32> {
        core::import_resources(&self.inner, resource as u32, name)
    }

    pub fn get_all_exported_resources(&self, resource: Resource) -> Vec<(String, u32)> {
        core::get_all_exported_resources(&self.inner, resource as u32)
    }

    pub fn release_exported_resources(&self, resource: Resource, name: &str) {
        core::release_exported_resources(&self.inner, resource as u32, name)
    }
}

/// --------------------------------------------------------------------------------

#[trait_variant::make(LocalRun: Send)]
pub trait Run {
    async fn run() -> Result<(), String>;
}

pub struct App<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> RunSync for App<T>
where
    T: Run,
{
    fn run() -> Result<(), String> {
        let result = wstd::runtime::block_on(async { T::run().await });
        if let Err(e) = result {
            return Err(format!("{:?}", e));
        }

        Ok(())
    }
}
#[trait_variant::make(LocalServe: Send)]
pub trait Serve {
    async fn serve(
        request: wstd::http::Request<wstd::http::body::IncomingBody>,
        responder: wstd::http::server::Responder,
    ) -> wstd::http::server::Finished;
}

pub struct Server<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> bindings_server::exports::wasi::http::incoming_handler::Guest for Server<T>
where
    T: Serve,
{
    fn handle(request: IncomingRequest, response_out: ResponseOutparam) -> () {
        let responder = wstd::http::server::Responder::new(response_out);
        let _finished: wstd::http::server::Finished =
            match wstd::http::request::try_from_incoming(request) {
                Ok(request) => {
                    ::wstd::runtime::block_on(async { T::serve(request, responder).await })
                }
                Err(err) => responder.fail(err),
            };
    }
}

#[macro_export]
macro_rules! main_sync {
    ($app:ident) => {
        pie::export!($app with_types_in pie::bindings_app);
    };
}

#[macro_export]
macro_rules! main_async {
    ($app:ident) => {
        type _App = pie::App<$app>;
        pie::export!(_App with_types_in pie::bindings_app);
    };
}

#[macro_export]
macro_rules! server {
    ($app:ident) => {
        type _Server = pie::Server<$app>;
        pie::wasi::http::proxy::export!(_Server with_types_in pie::bindings_server);
    };
}
