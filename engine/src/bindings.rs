mod l4m;
mod l4m_vision;
mod messaging;
mod ping;
mod runtime;

mod wit {
    use super::*;
    wasmtime::component::bindgen!({
        path: "../api/wit",
        world: "app",
        async: true,
        with: {
            "wasi:io/poll": wasmtime_wasi::bindings::io::poll,
            "symphony:app/messaging/subscription": messaging::Subscription,
            "symphony:app/l4m/model": l4m::Model,
            "symphony:app/l4m/tokenizer": l4m::Tokenizer,
            "symphony:app/l4m/sample-top-k-result": l4m::SampleTopKResult,
            "symphony:app/l4m/synchronization-result": l4m::SynchronizationResult,
        },
        trappable_imports: true,
    });
}

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: wit::symphony::app::l4m::Host
        + wit::symphony::app::l4m_vision::Host
        + wit::symphony::app::runtime::Host
        + wit::symphony::app::ping::Host
        + wit::symphony::app::messaging::Host,
{
    wit::symphony::app::l4m::add_to_linker(linker, |s| s)?;
    wit::symphony::app::l4m_vision::add_to_linker(linker, |s| s)?;
    wit::symphony::app::runtime::add_to_linker(linker, |s| s)?;
    wit::symphony::app::ping::add_to_linker(linker, |s| s)?;
    wit::symphony::app::messaging::add_to_linker(linker, |s| s)?;
    Ok(())
}
