use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__symphony_main_inner", original_fn_name.span());

    // Ensure the function is async
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[symphony::main] attribute can only be used on async functions",
        )
            .to_compile_error()
            .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `symphony::Run`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __SymphonyMain;

        impl symphony::RunSync for __SymphonyMain {
            fn run() -> Result<(), String> {
                let result = symphony::wstd::runtime::block_on(async { #inner_fn_name().await });
                if let Err(e) = result {
                    return Err(format!("{:?}", e));
                }
                Ok(())
            }
        }

        symphony::export!(__SymphonyMain with_types_in symphony::bindings_app);
    };

    expanded.into()
}
