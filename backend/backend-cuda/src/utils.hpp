#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <cstdlib> // For std::getenv
#include <cstring> // For std::strlen

namespace utils {
    // Basic Base64 decoding utility.
    inline std::vector<uint8_t> base64_decode(const std::string& in) {
        std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<int> T(256,-1);
        for (int i=0; i<64; i++) T[chars[i]] = i;

        std::vector<uint8_t> out;
        int val=0, valb=-8;
        for (unsigned char c : in) {
            // Handle padding characters if necessary, although the current implementation
            // just breaks on characters not in the base64 alphabet.
            if (T[c] == -1) {
                // If it's a padding character '=', we might want to ignore it or
                // handle it specifically. For now, breaking is acceptable for
                // a basic decoder that expects well-formed input without padding handling.
                if (c == '=') continue;
                break;
            }
            val = (val << 6) + T[c];
            valb += 6;
            if (valb >= 0) {
                out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return out;
    }

    // Implements a cross-platform method for finding the user's cache directory.
    inline std::filesystem::path get_user_cache_dir()
    {
#if defined(_WIN32)
        // Windows: Prefer %LOCALAPPDATA% for cache.
        const char *local_appdata = std::getenv("LOCALAPPDATA");
        if (local_appdata && std::strlen(local_appdata) > 0)
        {
            return std::filesystem::path(local_appdata);
        }
        // If LOCALAPPDATA is not set, which is highly unlikely on modern Windows,
        // we could fall back to APPDATA and append a common cache subdirectory.
        // However, it's generally better to rely on LOCALAPPDATA for cache.
        // For robustness, if LOCALAPPDATA is missing, we might throw or use a
        // temporary directory, but for cache, LOCALAPPDATA is the correct place.
        // The original code's APPDATA fallback was problematic.
        // A robust solution might involve getting the known folder path via SHGetKnownFolderPath,
        // but for environment variables, LOCALAPPDATA is key for cache.
        throw std::runtime_error("Could not determine local application data directory. Please ensure %LOCALAPPDATA% is set.");
#else
        // Linux, macOS, and other UNIX-like systems
        const char *home = std::getenv("HOME");
        if (!home || std::strlen(home) == 0)
        {
            throw std::runtime_error("Could not determine home directory. Please set the $HOME environment variable.");
        }
        std::filesystem::path home_path(home);

        // Use ~/.cache consistently across all platforms for simplicity
        // This ensures Docker volume mounts work identically across OSes
        return home_path / ".cache";
#endif
        // This throw statement is now unreachable due to the platform-specific #if/#else structure
        // ensuring a return or throw within each branch.
        // However, keeping it here as a safeguard if the platform detection were to fail in an unexpected way.
        throw std::runtime_error("Unsupported platform or unable to determine cache directory.");
    }
}

#endif // UTILS_HPP