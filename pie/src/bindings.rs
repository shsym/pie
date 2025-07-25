use wasmtime::component::HasSelf;

pub mod l4m;
mod l4m_vision;
pub mod messaging;
mod ping;
mod runtime;

mod wit {
    use super::*;
    wasmtime::component::bindgen!({
        path: "../inferlet/wit",
        world: "imports",
        async: true,
        with: {
            "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
            "pie:nbi/messaging/subscription": messaging::Subscription,
            "pie:nbi/messaging/receive-result": messaging::ReceiveResult,
            "pie:nbi/l4m/model": l4m::Model,
            "pie:nbi/l4m/tokenizer": l4m::Tokenizer,
            "pie:nbi/l4m/sample-top-k-result": l4m::SampleTopKResult,
            "pie:nbi/l4m/synchronization-result": l4m::SynchronizationResult,
        },
        trappable_imports: true,
    });
}

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: wit::pie::nbi::l4m::Host
        + wit::pie::nbi::l4m_vision::Host
        + wit::pie::nbi::runtime::Host
        + wit::pie::nbi::ping::Host
        + wit::pie::nbi::messaging::Host,
{
    wit::pie::nbi::l4m::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    wit::pie::nbi::l4m_vision::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    wit::pie::nbi::runtime::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    wit::pie::nbi::ping::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    wit::pie::nbi::messaging::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    Ok(())
}
