#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>

int main() {
    // Create Metal device and library
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "Failed to create Metal device" << std::endl;
        return 1;
    }
    
    // Load the default library from the app bundle
    NSString* libraryPath = @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/build/shaders/metal_batch_prefill_attention_flashattention.metal";
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithFile:libraryPath error:&error];
    
    if (!library) {
        std::cout << "Failed to load library: " << [[error localizedDescription] UTF8String] << std::endl;
        return 1;
    }
    
    // Try to load the FlashAttention function
    id<MTLFunction> function = [library newFunctionWithName:@"unified_batch_prefill_attention_bf16_flashattention"];
    if (!function) {
        std::cout << "FlashAttention kernel not found" << std::endl;
        
        // List all available functions in the library
        NSArray<NSString *> *functionNames = [library functionNames];
        std::cout << "Available functions:" << std::endl;
        for (NSString *name in functionNames) {
            std::cout << "  " << [name UTF8String] << std::endl;
        }
    } else {
        std::cout << "✅ FlashAttention kernel loaded successfully!" << std::endl;
    }
    
    return 0;
}