use wasmtime::component::HasSelf;

mod allocate;
mod forward;
mod input_image;
mod input_text;
mod model;
mod output_text;
mod runtime;
mod tokenize;

wasmtime::component::bindgen!({
    path: "../inferlet/wit2",
    world: "inferlet",
    async: true,
    with: {
        "wasi:io/poll": wasmtime_wasi::p2::bindings::io::poll,
        "pie:inferlet/runtime/subscription": runtime::Subscription,
        "pie:inferlet/runtime/receive-result": runtime::ReceiveResult,
        "pie:inferlet/model/model": model::Model,
        "pie:inferlet/model/queue": model::Queue,
        "pie:inferlet/model/synchronization-result": model::SynchronizationResult,
        "pie:inferlet/tokenize/tokenizer": tokenize::Tokenizer,
        "pie:inferlet/output-text/distribution-result": output_text::DistributionResult,
    },
    trappable_imports: true,
});

pub fn add_to_linker<T>(linker: &mut wasmtime::component::Linker<T>) -> Result<(), wasmtime::Error>
where
    T: pie::inferlet::model::Host
        + pie::inferlet::runtime::Host
        + pie::inferlet::allocate::Host
        + pie::inferlet::forward::Host
        + pie::inferlet::input_text::Host
        + pie::inferlet::input_image::Host
        + pie::inferlet::output_text::Host
        + pie::inferlet::tokenize::Host,
{
    pie::inferlet::model::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::runtime::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::allocate::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::forward::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::input_text::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::input_image::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::output_text::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    pie::inferlet::tokenize::add_to_linker::<T, HasSelf<T>>(linker, |s| s)?;
    Ok(())
}
