mod context;
pub mod drafter;
pub mod l4m_async;
pub mod messaging_async;
pub mod sampler;
pub mod stop_condition;
mod utils;

pub use symphony_macros::main;
pub use wstd;
pub mod bindings {
    wit_bindgen::generate!({
        path: "../../api/wit",
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
        path: "../../api/wit",
        world: "app",
        pub_export_macro: true,
        default_bindings_module: "symphony::bindings",
        with: {
             "wasi:io/poll@0.2.4": wasi::io::poll,
        },
    });
}

// pub mod bindings_server {
//     wit_bindgen::generate!({
//         path: "../../api/wit",
//         world: "server",
//         pub_export_macro: true,
//         default_bindings_module: "symphony::bindings",
//         with: {
//             "wasi:io/poll@0.2.4": wasi::io::poll,
//             "wasi:clocks/monotonic-clock@0.2.4": wasi::clocks::monotonic_clock,
//             "wasi:io/error@0.2.4": wasi::io::error,
//             "wasi:io/streams@0.2.4": wasi::io::streams,
//             "wasi:http/types@0.2.4": wasi::http::types,
//         },
//         generate_all,
//     });
// }

pub use crate::bindings::{
    symphony::nbi::l4m, symphony::nbi::l4m_vision, symphony::nbi::messaging, symphony::nbi::ping,
    symphony::nbi::runtime,
};

pub use crate::bindings_app::{export, exports::symphony::nbi::run::Guest as RunSync};

pub use crate::context::Model;
pub use anyhow::Result;
use wasi::exports::http::incoming_handler::{IncomingRequest, ResponseOutparam};

pub fn available_models() -> Vec<String> {
    Model::available_models()
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

impl<T> wasi::exports::http::incoming_handler::Guest for Server<T>
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
        symphony::export!($app with_types_in symphony::bindings);
    };
}

#[macro_export]
macro_rules! main_async {
    ($app:ident) => {
        type _App = symphony::App<$app>;
        symphony::export!(_App with_types_in symphony::bindings);
    };
}
