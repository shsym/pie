#pragma once

#include <filesystem>
#include <string>
#include <cstdlib>

namespace workspace_utils {

/**
 * Find the workspace root directory containing both cuda-protocol-tests and metal-protocol-tests
 * Returns empty path if not found
 */
inline std::filesystem::path find_workspace_root() {
    // First try environment variable if set
    const char* env_root = std::getenv("PIE_WORKSPACE_ROOT");
    if (env_root) {
        std::filesystem::path path(env_root);
        if (std::filesystem::exists(path / "cuda-protocol-tests") &&
            std::filesystem::exists(path / "metal-protocol-tests")) {
            return path;
        }
    }

    // Start from current executable directory and search upward
    std::filesystem::path current = std::filesystem::current_path();

    // Try relative to current working directory first
    while (!current.empty() && current != current.root_path()) {
        if (std::filesystem::exists(current / "cuda-protocol-tests") &&
            std::filesystem::exists(current / "metal-protocol-tests")) {
            return current;
        }
        current = current.parent_path();
    }

    // From backend-metal, the workspace root should be ../../../
    std::filesystem::path backend_relative = std::filesystem::current_path();
    for (int i = 0; i < 5; ++i) { // Search up to 5 levels up
        if (std::filesystem::exists(backend_relative / "cuda-protocol-tests") &&
            std::filesystem::exists(backend_relative / "metal-protocol-tests")) {
            return backend_relative;
        }
        backend_relative = backend_relative.parent_path();
        if (backend_relative.empty() || backend_relative == backend_relative.root_path()) {
            break;
        }
    }

    return {}; // Not found
}

/**
 * Get the absolute path to the CUDA artifacts directory
 */
inline std::filesystem::path get_cuda_artifacts_dir() {
    // First check environment variable
    const char* env_path = std::getenv("PIE_CUDA_ARTIFACTS_DIR");
    if (env_path) {
        return std::filesystem::path(env_path);
    }

    // Find workspace root and construct path
    auto workspace_root = find_workspace_root();
    if (workspace_root.empty()) {
        // Return empty path to indicate failure
        return {};
    }

    // Prefer metal-protocol-tests location (where CUDA artifacts are generated)
    std::filesystem::path cuda_path = workspace_root / "cuda-protocol-tests" / "tests" / "artifacts";
    return cuda_path;
}

/**
 * Get the absolute path to the Metal artifacts directory
 */
inline std::filesystem::path get_metal_artifacts_dir() {
    auto workspace_root = find_workspace_root();
    if (workspace_root.empty()) {
        // Return empty path to indicate failure
        return {};
    }
    return workspace_root / "metal-protocol-tests" / "tests" / "artifacts";
}

} // namespace workspace_utils