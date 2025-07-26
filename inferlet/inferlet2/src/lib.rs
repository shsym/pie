mod command;
pub mod context;
pub mod core_async;
pub mod drafter;
pub mod output_text_async;
mod pool;
pub mod sampler;
pub mod stop_condition;
mod traits;

pub use inferlet_macros2::main;
pub use inferlet_macros2::server_main;
use std::rc::Rc;

pub use wstd;
pub mod bindings {
    wit_bindgen::generate!({
        path: "../wit2",
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
        path: "../wit2",
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
        path: "../wit2",
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

pub use crate::bindings::pie::inferlet::{
    allocate, core, forward, input_image, input_text, output_text, tokenize,
};

pub use crate::bindings_app::{export, exports::pie::inferlet::run::Guest as RunSync};
pub use crate::context::Context;
pub use anyhow::Result;
pub use wasi;
use wasi::exports::http::incoming_handler::{IncomingRequest, ResponseOutparam};

#[derive(Clone, Debug)]
pub struct Queue {
    pub(crate) inner: Rc<core::Queue>,
}

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
