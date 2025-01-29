mod spi;
use futures::executor::block_on;

use crate::spi::{App, ComponentRunStates};
use anyhow::{anyhow, Context};
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Result, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiImpl, WasiView};

fn main() -> Result<()> {
    let mut config = Config::default();
    config.async_support(true);

    let engine = Engine::new(&config)?;

    // run_async(
    //     &engine,
    //     "../example-apps/target/wasm32-wasip2/release/helloworld.wasm",
    // )?;

    run_async(
        &engine,
        "../example-apps-python/helloworld/helloworld.wasm",
    )?;
    
    println!("Run without errors!");
    Ok(())
}

fn run_async(engine: &Engine, path: &'static str) -> Result<()> {
    // get component and linker
    let component = Component::from_file(engine, path)
        .with_context(|| format!("Cannot find component from path: {path}"))?;

    let mut linker: Linker<ComponentRunStates> = Linker::new(&engine);
    let state = ComponentRunStates::new();
    let mut store = Store::new(&engine, state);

    App::add_to_linker(&mut linker, |s| s)?;
    bind_interfaces_needed_by_guest_rust_std(&mut linker);
    let async_future = async {
        let instance = linker.instantiate_async(&mut store, &component).await?;

        let run_interface = instance
            .get_export(&mut store, None, "spi:app/run")
            .ok_or_else(|| anyhow!("spi:app/run missing?"))?;
        let run_func_export = instance
            .get_export(&mut store, Some(&run_interface), "run")
            .ok_or_else(|| anyhow!("run export missing?"))?;
        let run_func = instance
            .get_typed_func::<(), (Result<(), ()>,)>(&mut store, &run_func_export)
            .context("run as typed func")?;

        println!("entering wasm...");
        let (runtime_result,) = run_func.call_async(&mut store, ()).await?;
        runtime_result.map_err(|()| anyhow!("run returned an error"))?;
        println!("done");
        Ok(())
    };
    block_on(async_future)
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
