#include "handshake.pb.h"
#include "l4m.pb.h"

#include <iostream>
#include <string>

#include <zmq.hpp>
#include <nlohmann/json.hpp>

int main() {
    // Print hello world
    std::cout << "Hello, World!" << std::endl;

    // Create a handshake::Response message
    handshake::Response response;
    response.add_protocols("http/1.1");
    response.add_protocols("spdy/3.1");
    response.add_protocols("h2");

    // Serialize the message to a string
    std::string serialized;
    if (!response.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize response." << std::endl;
        return 1;
    }

    std::cout << "Serialized Response size: " << serialized.size() << " bytes" << std::endl;

    // Deserialize the message back
    handshake::Response deserialized;
    if (!deserialized.ParseFromString(serialized)) {
        std::cerr << "Failed to deserialize response." << std::endl;
        return 1;
    }

    // Print protocols
    std::cout << "Deserialized Protocols:" << std::endl;
    for (const auto& proto : deserialized.protocols()) {
        std::cout << "- " << proto << std::endl;
    }

    return 0;
}