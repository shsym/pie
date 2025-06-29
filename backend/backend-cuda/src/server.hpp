#pragma once

#include "config.hpp"
#include <memory>

// The PIMPL Idiom: The Server class will manage the application's lifecycle.
// It hides all implementation details (like the model pointer) behind an
// opaque pointer, breaking the compile-time dependency.

class Server {
public:
    /**
     * @brief Constructs the server, which includes loading the model.
     * @param config The application configuration.
     * @param out_metadata A reference that will be populated with the model's metadata.
     */
    Server(const AppConfig& config, ModelMetadata& out_metadata);

    /**
     * @brief Destructor. MUST be defined in the .cu file.
     */
    ~Server();

    /**
     * @brief Starts the main server loop to listen for requests.
     */
    void run();

private:
    // Forward-declare the implementation struct. Its full definition
    // will only be visible inside server.cu.
    struct ServerImpl;

    // The opaque pointer to the implementation.
    std::unique_ptr<ServerImpl> pimpl;
};
