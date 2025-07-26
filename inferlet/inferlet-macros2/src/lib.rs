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
            "The #[inferlet2::main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `inferlet2::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl inferlet2::RunSync for __PieMain {
            fn run() -> Result<(), String> {
                let result = inferlet2::wstd::runtime::block_on(async { #inner_fn_name().await });
                if let Err(e) = result {
                    return Err(format!("{:?}", e));
                }
                Ok(())
            }
        }

        inferlet2::export!(__PieMain with_types_in inferlet2::bindings_app);
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
            "The #[inferlet2::server_main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `inferlet2::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl inferlet2::bindings_server::exports::wasi::http::incoming_handler::Guest for __PieMain {
            fn handle(request: inferlet2::wasi::exports::http::incoming_handler::IncomingRequest, response_out: inferlet2::wasi::exports::http::incoming_handler::ResponseOutparam) -> () {
                let responder = inferlet2::wstd::http::server::Responder::new(response_out);
                let _finished: inferlet2::wstd::http::server::Finished =
                    match inferlet2::wstd::http::request::try_from_incoming(request) {
                        Ok(request) => {
                            inferlet2::wstd::runtime::block_on(async { #inner_fn_name(request, responder).await })
                        }
                        Err(err) => responder.fail(err),
                    };
            }
        }

        inferlet2::wasi::http::proxy::export!(__PieMain with_types_in inferlet2::bindings_server);
    };

    expanded.into()
}
