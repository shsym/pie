use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__pie_main_inner", original_fn_name.span());

    // Ensure the function is async
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[pie::main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `pie::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl pie::RunSync for __PieMain {
            fn run() -> Result<(), String> {
                let result = pie::wstd::runtime::block_on(async { #inner_fn_name().await });
                if let Err(e) = result {
                    return Err(format!("{:?}", e));
                }
                Ok(())
            }
        }

        pie::export!(__PieMain with_types_in pie::bindings_app);
    };

    expanded.into()
}



#[proc_macro_attribute]
pub fn server_main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__pie_main_inner", original_fn_name.span());

    // Ensure the function is async
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[pie::server_main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `pie::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl pie::bindings_server::exports::wasi::http::incoming_handler::Guest for __PieMain {
            fn handle(request: pie::wasi::exports::http::incoming_handler::IncomingRequest, response_out: pie::wasi::exports::http::incoming_handler::ResponseOutparam) -> () {
                let responder = pie::wstd::http::server::Responder::new(response_out);
                let _finished: pie::wstd::http::server::Finished =
                    match pie::wstd::http::request::try_from_incoming(request) {
                        Ok(request) => {
                            pie::wstd::runtime::block_on(async { #inner_fn_name(request, responder).await })
                        }
                        Err(err) => responder.fail(err),
                    };
            }
        }

        pie::wasi::http::proxy::export!(__PieMain with_types_in pie::bindings_server);
    };

    expanded.into()
}
