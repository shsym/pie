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
            "The #[inferlet::main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `inferlet::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl inferlet::RunSync for __PieMain {
            fn run() -> Result<(), String> {
                let result = inferlet::wstd::runtime::block_on(async { #inner_fn_name().await });
                if let Err(e) = result {
                    return Err(format!("{:?}", e));
                }
                Ok(())
            }
        }

        inferlet::export!(__PieMain with_types_in inferlet::bindings_app);
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
            "The #[inferlet::server_main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `inferlet::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl inferlet::bindings_server::exports::wasi::http::incoming_handler::Guest for __PieMain {
            fn handle(request: inferlet::wasi::exports::http::incoming_handler::IncomingRequest, response_out: inferlet::wasi::exports::http::incoming_handler::ResponseOutparam) -> () {
                let responder = inferlet::wstd::http::server::Responder::new(response_out);
                let _finished: inferlet::wstd::http::server::Finished =
                    match inferlet::wstd::http::request::try_from_incoming(request) {
                        Ok(request) => {
                            inferlet::wstd::runtime::block_on(async { #inner_fn_name(request, responder).await })
                        }
                        Err(err) => responder.fail(err),
                    };
            }
        }

        inferlet::wasi::http::proxy::export!(__PieMain with_types_in inferlet::bindings_server);
    };

    expanded.into()
}
