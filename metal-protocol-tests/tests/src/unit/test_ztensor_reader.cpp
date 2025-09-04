#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <cassert>
#include <filesystem>

// Include the CUDA backend's ztensor implementation
#include "../../../../backend/backend-cuda/src/ztensor.hpp"

namespace fs = std::filesystem;

class ZTensorReaderTest {
public:
    void run_all_tests() {
        std::cout << "Running zTensor Reader Tests...\n";

        setup();

        test_basic_functionality();
        test_invalid_file();
        test_invalid_magic();
        test_file_too_small();
        test_memory_mapping();
        test_system_endianness();

        cleanup();

        std::cout << "✅ All zTensor Reader tests passed!\n";
    }

private:
    std::string test_data_path;

    void setup() {
        test_data_path = fs::temp_directory_path() / "ztensor_test";
        fs::create_directories(test_data_path);
        std::cout << "Test data path: " << test_data_path << "\n";
    }

    void cleanup() {
        if (fs::exists(test_data_path)) {
            fs::remove_all(test_data_path);
        }
    }

    void create_test_ztensor_file(const std::string& filename, int tensor_count) {
        std::string filepath = test_data_path + "/" + filename;
        std::ofstream file(filepath, std::ios::binary);

        // Write magic number
        file.write("ZTEN0001", 8);

        if (tensor_count == 0) {
            // Empty tensor file
            uint8_t empty_cbor = 0x80; // Empty CBOR array
            file.write(reinterpret_cast<const char*>(&empty_cbor), 1);

            uint64_t cbor_size = 1;
            file.write(reinterpret_cast<const char*>(&cbor_size), sizeof(cbor_size));
            file.close();
            return;
        }

        // For non-empty files, we'd need proper CBOR encoding
        // For now, just create a minimal valid file structure
        uint8_t empty_cbor = 0x80; // Empty CBOR array (no tensors for simplicity)
        file.write(reinterpret_cast<const char*>(&empty_cbor), 1);

        uint64_t cbor_size = 1;
        file.write(reinterpret_cast<const char*>(&cbor_size), sizeof(cbor_size));

        file.close();
    }

    void test_basic_functionality() {
        std::cout << "Testing basic functionality...\n";

        create_test_ztensor_file("empty_test.ztensor", 0);
        std::string filepath = test_data_path + "/empty_test.ztensor";

        try {
            ztensor::zTensorReader reader(filepath);

            // Test basic methods
            auto tensor_list = reader.list_tensors();
            assert(tensor_list.size() == 0);

            std::cout << "  ✅ Basic functionality test passed\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Exception in basic functionality test: " << e.what() << std::endl;
            throw;
        }
    }

    void test_invalid_file() {
        std::cout << "Testing invalid file handling...\n";

        std::string filepath = test_data_path + "/nonexistent.ztensor";

        bool exception_thrown = false;
        try {
            ztensor::zTensorReader reader(filepath);
        } catch (const std::exception& e) {
            exception_thrown = true;
        }

        assert(exception_thrown);
        std::cout << "  ✅ Invalid file test passed\n";
    }

    void test_invalid_magic() {
        std::cout << "Testing invalid magic number...\n";

        std::string filepath = test_data_path + "/invalid_magic.ztensor";
        std::ofstream file(filepath, std::ios::binary);
        file.write("INVALID1", 8); // Wrong magic
        uint8_t empty_cbor = 0x80;
        file.write(reinterpret_cast<const char*>(&empty_cbor), 1);
        uint64_t cbor_size = 1;
        file.write(reinterpret_cast<const char*>(&cbor_size), sizeof(cbor_size));
        file.close();

        bool exception_thrown = false;
        try {
            ztensor::zTensorReader reader(filepath);
        } catch (const std::exception& e) {
            exception_thrown = true;
        }

        assert(exception_thrown);
        std::cout << "  ✅ Invalid magic number test passed\n";
    }

    void test_file_too_small() {
        std::cout << "Testing file too small...\n";

        std::string filepath = test_data_path + "/too_small.ztensor";
        std::ofstream file(filepath, std::ios::binary);
        file.write("ZTEN0001", 8); // Just magic, nothing else
        file.close();

        bool exception_thrown = false;
        try {
            ztensor::zTensorReader reader(filepath);
        } catch (const std::exception& e) {
            exception_thrown = true;
        }

        assert(exception_thrown);
        std::cout << "  ✅ File too small test passed\n";
    }

    void test_memory_mapping() {
        std::cout << "Testing memory mapping...\n";

        create_test_ztensor_file("mapping_test.ztensor", 0);
        std::string filepath = test_data_path + "/mapping_test.ztensor";

        try {
            ztensor::zTensorReader reader(filepath);

            // If we got here without exception, memory mapping worked
            auto tensor_list = reader.list_tensors();
            assert(tensor_list.size() == 0);

            std::cout << "  ✅ Memory mapping test passed\n";

        } catch (const std::exception& e) {
            std::cerr << "  ❌ Memory mapping should work for valid file: " << e.what() << std::endl;
            throw;
        }
    }

    void test_system_endianness() {
        std::cout << "Testing system endianness detection...\n";

        bool is_little_endian = ztensor::is_system_little_endian();

        // On Apple Silicon and Intel Macs, we should have little-endian
        assert(is_little_endian);
        std::cout << "  ✅ System endianness test passed (little-endian detected)\n";
    }
};

int main() {
    try {
        ZTensorReaderTest test;
        test.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}