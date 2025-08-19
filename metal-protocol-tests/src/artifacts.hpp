#pragma once

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
#include <cctype>

namespace artifacts {

inline bool get_env_flag(const char* name, bool default_val = false) {
    const char* v = std::getenv(name);
    if (!v) return default_val;
    std::string s(v);
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
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
    // Expect PIE_ARTIFACTS_DIR to be set by the harness (defaults to a path inside build dir).
    // Fall back to a local tests/artifacts if not provided (main.cpp sets it early).
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

// Convenience for writing host memory buffers (pointer + count)
template <typename T>
inline void write_host_bin(const std::filesystem::path& dir, std::string_view name, const T* ptr, size_t count) {
    write_bytes(dir / (std::string(name) + ".bin"), ptr, count * sizeof(T));
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

} // namespace artifacts
