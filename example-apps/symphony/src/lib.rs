mod context;
pub mod drafter;
pub mod l4m_async;
pub mod sampler;
pub mod stop_condition;
mod utils;

pub use wstd;
pub mod bindings {
    wit_bindgen::generate!({
        path: "../../api/wit",
        world: "app",
        pub_export_macro: true,
        export_macro_name: "export",
        with: {
             "wasi:io/poll@0.2.4": wasi::io::poll,
        },
        generate_all,
    });
}

pub use crate::bindings::{
    export, exports::symphony::app::run::Guest as RunSync, symphony::app::l4m,
    symphony::app::l4m_vision, symphony::app::ping, symphony::app::runtime, symphony::app::messaging
};
pub use crate::context::Model;

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

        // let runtime = tokio::runtime::Builder::new_current_thread()
        //     .enable_all()
        //     .build()
        //     .unwrap();
        //
        // let local = tokio::task::LocalSet::new();
        // let result = local.block_on(&runtime, T::run());
        // Run the async main function inside the runtime
        //let result = runtime.block_on(T::run());

        if let Err(e) = result {
            return Err(format!("{:?}", e));
        }

        Ok(())
    }
}

#[macro_export]
macro_rules! main_sync {
    ($app:ident) => {
        symphony::export!($app with_types_in symphony::bindings);
    };
}

#[macro_export]
macro_rules! main {
    ($app:ident) => {
        type _App = symphony::App<$app>;
        symphony::export!(_App with_types_in symphony::bindings);
    };
}
