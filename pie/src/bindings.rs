use wasmtime::component::HasSelf;

mod allocate;
mod core;
mod forward;
mod input_image;
mod input_text;
mod output_text;
//mod runtime;
mod forward_text;
mod tokenize;
mod optimize;

wasmtime::component::bindgen!({
    path: "wit",
    world: "inferlet",
    async: true,
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        "pie:inferlet/core/subscription": core::Subscription,
        "pie:inferlet/core/receive-result": core::ReceiveResult,
        "pie:inferlet/core/model": core::Model,
        "pie:inferlet/core/queue": core::Queue,
        "pie:inferlet/core/debug-query-result": core::DebugQueryResult,
        "pie:inferlet/core/synchronization-result": core::SynchronizationResult,
        "pie:inferlet/tokenize/tokenizer": tokenize::Tokenizer,
        "pie:inferlet/output-text/distribution-result": output_text::DistributionResult,
        "pie:inferlet/forward-text/distribution-result": forward_text::DistributionResult,
    },
    trappable_imports: true,
});

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: pie::inferlet::core::Host
        + pie::inferlet::allocate::Host
        + pie::inferlet::forward::Host
        + pie::inferlet::forward_text::Host
        + pie::inferlet::input_text::Host
        + pie::inferlet::input_image::Host
        + pie::inferlet::output_text::Host
        + pie::inferlet::tokenize::Host
        + pie::inferlet::optimize::Host,
{ 
    pie::inferlet::core::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::allocate::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::forward::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::forward_text::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::input_text::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::input_image::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::output_text::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::tokenize::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::optimize::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    Ok(())
}
