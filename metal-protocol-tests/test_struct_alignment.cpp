#include <iostream>
#include <cstdint>

// Replicate the exact struct used in the Metal code
struct Params {
    int num_qo;
    int head_dim;
    int head_size;
    int page_size;
    float scale;
};

int main() {
    std::cout << "C++ struct analysis:" << std::endl;
    std::cout << "sizeof(int): " << sizeof(int) << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << std::endl;
    std::cout << "sizeof(Params): " << sizeof(Params) << std::endl;
    
    Params p = { 128, 4096, 128, 16, 0.0883883f };
    
    std::cout << "\nField offsets:" << std::endl;
    std::cout << "num_qo@" << ((char*)&p.num_qo - (char*)&p) << std::endl;
    std::cout << "head_dim@" << ((char*)&p.head_dim - (char*)&p) << std::endl;
    std::cout << "head_size@" << ((char*)&p.head_size - (char*)&p) << std::endl;
    std::cout << "page_size@" << ((char*)&p.page_size - (char*)&p) << std::endl;
    std::cout << "scale@" << ((char*)&p.scale - (char*)&p) << std::endl;
    
    std::cout << "\nRaw bytes:" << std::endl;
    uint8_t* raw = (uint8_t*)&p;
    for (size_t i = 0; i < sizeof(Params); ++i) {
        std::printf("%02x ", raw[i]);
        if ((i + 1) % 8 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "\nField values:" << std::endl;
    std::cout << "p.num_qo = " << p.num_qo << std::endl;
    std::cout << "p.head_dim = " << p.head_dim << std::endl;
    std::cout << "p.head_size = " << p.head_size << std::endl;
    std::cout << "p.page_size = " << p.page_size << std::endl;
    std::cout << "p.scale = " << p.scale << std::endl;
    
    return 0;
}