mod l4m;
mod l4m_vision;
mod messaging;
mod ping;
mod runtime;

mod wit {
    use super::*;
    wasmtime::component::bindgen!({
        path: "../api/wit",
        world: "imports",
        async: true,
        with: {
            "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
            "symphony:nbi/messaging/subscription": messaging::Subscription,
            "symphony:nbi/messaging/receive-result": messaging::ReceiveResult,
            "symphony:nbi/l4m/model": l4m::Model,
            "symphony:nbi/l4m/tokenizer": l4m::Tokenizer,
            "symphony:nbi/l4m/sample-top-k-result": l4m::SampleTopKResult,
            "symphony:nbi/l4m/synchronization-result": l4m::SynchronizationResult,
        },
        trappable_imports: true,
    });
}

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: wit::symphony::nbi::l4m::Host
        + wit::symphony::nbi::l4m_vision::Host
        + wit::symphony::nbi::runtime::Host
        + wit::symphony::nbi::ping::Host
        + wit::symphony::nbi::messaging::Host,
{
    wit::symphony::nbi::l4m::add_to_linker(linker, |s| s)?;
    wit::symphony::nbi::l4m_vision::add_to_linker(linker, |s| s)?;
    wit::symphony::nbi::runtime::add_to_linker(linker, |s| s)?;
    wit::symphony::nbi::ping::add_to_linker(linker, |s| s)?;
    wit::symphony::nbi::messaging::add_to_linker(linker, |s| s)?;
    Ok(())
}
