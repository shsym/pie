
/// Raw layer with unsafe no-mangle functions for WASM runtime
mod raw {
    #[no_mangle]
    pub extern "C" fn tokenize(input_ptr: *const u8, input_len: usize) -> u32 {
        unimplemented!("This function will be implemented by the WASM runtime");
    }

    #[no_mangle]
    pub extern "C" fn pred(token_id: u32) -> u32 {
        unimplemented!("This function will be implemented by the WASM runtime");
    }

    #[no_mangle]
    pub extern "C" fn top(prediction_id: u32) -> u32 {
        unimplemented!("This function will be implemented by the WASM runtime");
    }

    #[no_mangle]
    pub extern "C" fn detokenize(token_id: u32, output_ptr: *mut u8, output_len: usize) -> usize {
        unimplemented!("This function will be implemented by the WASM runtime");
    }

    #[no_mangle]
    pub extern "C" fn output(data_ptr: *const u8, data_len: usize) {
        unimplemented!("This function will be implemented by the WASM runtime");
    }
}



/// Safe API exposed to the users

pub mod lm {
    use crate::raw;
    /// Tokenizes a given input string.
    pub fn tokenize(input: &str) -> u32 {
        // Safe call to the underlying raw function
        unsafe { raw::tokenize(input.as_ptr(), input.len()) }
    }

    /// Generates a prediction ID based on a tokenized input.
    pub fn pred(token_id: u32) -> u32 {
        // Safe call to the underlying raw function
        unsafe { raw::pred(token_id) }
    }

    /// Samples the top result from a prediction.
    pub fn top(prediction_id: u32) -> u32 {
        // Safe call to the underlying raw function
        unsafe { raw::top(prediction_id) }
    }

    /// Detokenizes a token ID into a string output.
    pub fn detokenize(token_id: u32) -> String {
        // Allocate a buffer for the result
        let mut buffer = vec![0u8; 1024]; // 1 KB buffer for simplicity
        let len = unsafe { raw::detokenize(token_id, buffer.as_mut_ptr(), buffer.len()) };

        // Ensure valid UTF-8 output
        String::from_utf8(buffer[..len].to_vec()).expect("Invalid UTF-8 in detokenized output")
    }
}

pub mod io {
    use crate::raw;
    /// Outputs data as a string
    pub fn output(data: &str) {
        unsafe {
            raw::output(data.as_ptr(), data.len());
        }
    }
}
