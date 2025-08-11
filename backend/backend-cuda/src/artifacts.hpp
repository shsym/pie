#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <iomanip>

namespace artifacts {

inline bool get_env_flag(const char* name, bool default_val = false) {
    const char* v = std::getenv(name);
    if (!v) return default_val;
    std::string s(v);
    for (auto& c : s) c = std::tolower(c);
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

inline std::string get_env_str(const char* name, std::string fallback = {}) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : fallback;
}

inline bool list_contains(std::string_view csv_or_ssv, std::string_view needle) {
    size_t start = 0;
    while (start < csv_or_ssv.size()) {
        size_t end = csv_or_ssv.find_first_of(",;:", start);
        if (end == std::string_view::npos) end = csv_or_ssv.size();
        auto token = csv_or_ssv.substr(start, end - start);
        // trim spaces
        size_t l = 0, r = token.size();
        while (l < r && std::isspace(static_cast<unsigned char>(token[l]))) ++l;
        while (r > l && std::isspace(static_cast<unsigned char>(token[r - 1]))) --r;
        if (needle == token.substr(l, r - l)) return true;
        start = end + 1;
    }
    return false;
}

inline bool op_enabled(std::string_view op_name) {
    if (!get_env_flag("PIE_WRITE_ARTIFACTS", false)) return false;
    std::string list = get_env_str("PIE_ARTIFACT_OPS", "");
    if (list.empty()) return true; // all ops enabled
    return list_contains(list, op_name);
}

inline std::filesystem::path ensure_dir_for_case(std::string_view op, std::string_view case_id) {
    std::filesystem::path base = get_env_str("PIE_ARTIFACTS_DIR", "tests/artifacts");
    std::filesystem::path dir = base / std::string(op) / std::string(case_id);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    (void)ec; // ignore errors; will surface on file open
    return dir;
}

inline void write_bytes(const std::filesystem::path& file, const void* data, size_t nbytes) {
    std::ofstream ofs(file, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
}

template <typename T>
inline void write_vector_bin(const std::filesystem::path& dir, std::string_view name, const std::vector<T>& v) {
    write_bytes(dir / (std::string(name) + ".bin"), v.data(), v.size() * sizeof(T));
}

template <typename T>
inline void write_device_bin(const std::filesystem::path& dir, std::string_view name, const T* dptr, size_t count) {
    std::vector<T> h(count);
    if (count > 0) {
        cudaMemcpy(h.data(), dptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    write_vector_bin(dir, name, h);
}

inline void write_text(const std::filesystem::path& file, const std::string& s) {
    std::ofstream ofs(file);
    ofs << s;
}

inline std::string json_escape(const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (auto c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    o << "\\u";
                    const char* hex = "0123456789abcdef";
                    o << '0' << '0' << hex[(c >> 4) & 0xF] << hex[c & 0xF];
                } else {
                    o << c;
                }
        }
    }
    o << '"';
    return o.str();
}

// Minimal meta writer; caller builds the JSON body content string
inline void write_meta_json(const std::filesystem::path& dir, const std::string& json_body) {
    std::ostringstream os;
    os << "{\n" << json_body << "\n}\n";
    write_text(dir / "meta.json", os.str());
}

// Format a 64-bit value as zero-padded lowercase hex (16 nybbles)
inline std::string to_hex64(uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::setfill('0') << std::setw(16) << std::nouppercase << v;
    return os.str();
}

// FNV-1a 64-bit hash over a byte span
inline uint64_t fnv1a64_bytes(const uint8_t* data, size_t n) {
    constexpr uint64_t offset_basis = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;
    uint64_t hash = offset_basis;
    for (size_t i = 0; i < n; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= prime;
    }
    return hash;
}

// Compute FNV-1a 64-bit hash of device memory interpreted as bytes of T[count]
template <typename T>
inline uint64_t compute_device_fnv1a64(const T* dptr, size_t count) {
    if (count == 0) {
        // Return FNV-1a offset basis for empty input to match fnv1a64_bytes semantics
        return 14695981039346656037ull;
    }
    std::vector<T> h(count);
    cudaMemcpy(h.data(), dptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    return fnv1a64_bytes(reinterpret_cast<const uint8_t*>(h.data()), count * sizeof(T));
}

// Write a flat JSON object of string -> string pairs
inline void write_json_object(const std::filesystem::path& file, const std::vector<std::pair<std::string, std::string>>& kvs) {
    std::ostringstream os;
    os << "{\n";
    for (size_t i = 0; i < kvs.size(); ++i) {
        os << "  " << json_escape(kvs[i].first) << ": " << json_escape(kvs[i].second);
        if (i + 1 < kvs.size()) os << ",";
        os << "\n";
    }
    os << "}\n";
    write_text(file, os.str());
}

} // namespace artifacts
