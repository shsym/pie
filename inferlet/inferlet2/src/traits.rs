pub mod allocate;
pub mod forward;
mod forward_text;
pub mod input_image;
pub mod input_text;
pub mod output_text;
pub mod tokenize;

// reexport traits
pub use allocate::Allocate;
pub use forward::Forward;
pub use forward_text::ForwardText;
pub use input_image::InputImage;
pub use input_text::InputText;
pub use output_text::OutputText;
pub use tokenize::Tokenize;
