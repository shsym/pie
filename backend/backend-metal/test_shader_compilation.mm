#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>

int main() {
    // Create Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "Failed to create Metal device" << std::endl;
        return 1;
    }
    
    // Load FlashAttention shader source
    NSString* shaderPath = @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_batch_prefill_attention_flashattention.metal";
    NSError* error = nil;
    
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath 
                                                       encoding:NSUTF8StringEncoding 
                                                          error:&error];
    
    if (error || !shaderSource) {
        std::cout << "Failed to load shader source: " << (error ? error.localizedDescription.UTF8String : "unknown") << std::endl;
        return 1;
    }
    
    std::cout << "Shader source loaded: " << shaderSource.length << " characters" << std::endl;
    
    // Try to compile the shader
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeFast;
    } else {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
        #pragma clang diagnostic pop
    }
    
    std::cout << "Compiling Metal shader..." << std::endl;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:options error:&error];
    
    if (!library) {
        std::cout << "❌ Shader compilation failed: " << (error ? error.localizedDescription.UTF8String : "unknown") << std::endl;
        return 1;
    }
    
    std::cout << "✅ Shader compiled successfully!" << std::endl;
    
    // Try to load the FlashAttention function
    id<MTLFunction> function = [library newFunctionWithName:@"unified_batch_prefill_attention_bf16_flashattention"];
    if (!function) {
        std::cout << "❌ FlashAttention kernel function not found" << std::endl;
        
        // List all available functions
        NSArray<NSString *> *functionNames = [library functionNames];
        std::cout << "Available functions:" << std::endl;
        for (NSString *name in functionNames) {
            std::cout << "  " << [name UTF8String] << std::endl;
        }
        return 1;
    }
    
    std::cout << "✅ FlashAttention kernel function loaded successfully!" << std::endl;
    return 0;
}