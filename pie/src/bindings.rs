use wasmtime::component::HasSelf;

use crate::handler::{adapter, core, evolve, forward, image, tokenize};

wasmtime::component::bindgen!({
    path: "wit",
    world: "inferlet",
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        "pie:inferlet/core/subscription": core::Subscription,
        "pie:inferlet/core/receive-result": core::ReceiveResult,
        "pie:inferlet/core/model": core::Model,
        "pie:inferlet/core/queue": core::Queue,
        "pie:inferlet/core/debug-query-result": core::DebugQueryResult,
        "pie:inferlet/core/synchronization-result": core::SynchronizationResult,
        "pie:inferlet/forward/forward-pass": forward::ForwardPass,
        "pie:inferlet/forward/forward-pass-result": forward::ForwardPassResult,
        "pie:inferlet/tokenize/tokenizer": tokenize::Tokenizer,
    },
    imports: { default: async | trappable },
    exports: { default: async },
});

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: pie::inferlet::core::Host
        + pie::inferlet::forward::Host
        + pie::inferlet::adapter::Host
        + pie::inferlet::evolve::Host
        + pie::inferlet::image::Host
        + pie::inferlet::tokenize::Host,
{
    pie::inferlet::core::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::forward::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::adapter::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::evolve::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::image::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::tokenize::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    Ok(())
}
