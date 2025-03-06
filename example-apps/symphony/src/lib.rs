mod context;
pub mod sampler;
pub mod stop_condition;

pub use context::Context;

pub mod bindings {

    wit_bindgen::generate!({
        path: "../../api/wit",
        world: "app",
        pub_export_macro: true,
        export_macro_name: "export",
        generate_all,
    });
}

pub use crate::bindings::{
    export, exports::spi::app::run::Guest as RunSync, spi::app::l4m, spi::app::l4m_vision,
    spi::app::ping, spi::app::system,
};

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
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        // Run the async main function inside the runtime
        let result = runtime.block_on(T::run());

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
