mod spi;

use crate::spi::{ComponentRunStates, Imports};
use anyhow::Context;
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Result, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiImpl, WasiView};

fn main() -> Result<()> {
    let mut config = Config::default();
    config.async_support(true);

    let engine = Engine::new(&config)?;

    println!("Run without errors!");
    Ok(())
}

fn run_async(engine: &Engine, path: &'static str) -> Result<()> {
    // get component and linker
    let component = Component::from_file(engine, path)
        .with_context(|| format!("Cannot find component from path: {path}"))?;

    let mut linker = Linker::new(&engine);
    let state = ComponentRunStates::new();
    let mut store = Store::new(&engine, state);

    Imports::add_to_linker(&mut linker, |s| s)?;
    bind_interfaces_needed_by_guest_rust_std(&mut linker);

    // let async_future = async {
    //     let bindings = KvDatabase::instantiate_async(&mut store, &component, &linker).await?;
    //     let result = bindings.call_replace_value(store, "hello", "world").await?;
    //     assert_eq!(result, None);
    //     Ok(())
    // };
    // block_on(async_future)
    //
    // unimplemented!("Run async code here")
}

/// Copied from [wasmtime_wasi::type_annotate]
pub fn type_annotate<T: WasiView, F>(val: F) -> F
where
    F: Fn(&mut T) -> WasiImpl<&mut T>,
{
    val
}
pub fn bind_interfaces_needed_by_guest_rust_std<T: WasiView>(l: &mut Linker<T>) {
    let closure = type_annotate::<T, _>(|t| WasiImpl(t));
    let options = wasmtime_wasi::bindings::sync::LinkOptions::default();
    wasmtime_wasi::bindings::sync::filesystem::types::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::filesystem::preopens::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::io::error::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::sync::io::streams::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::cli::exit::add_to_linker_get_host(l, &options.into(), closure)
        .unwrap();
    wasmtime_wasi::bindings::cli::environment::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::cli::stdin::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::cli::stdout::add_to_linker_get_host(l, closure).unwrap();
    wasmtime_wasi::bindings::cli::stderr::add_to_linker_get_host(l, closure).unwrap();
}
