#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <sstream>

// Forward declarations
class LoggingManager;
class PerformanceLogger;

// Helper macro to safely check CUDA calls
#define CUDA_CHECK_PROFILER(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        char error_buf[256]; \
        snprintf(error_buf, sizeof(error_buf), "CUDA Error in Profiler at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(error_buf); \
    } \
} while (0)

// A node in the performance report tree
struct ProfileNode {
    double mean_ms = 0.0;
    double std_dev = 0.0;
    size_t count = 0;
    double total_mean_ms = 0.0;
    std::map<std::string, ProfileNode> children;
};

/**
 * @class PerformanceLogger
 * @brief Manages a nested, independent timing scope.
 * Reports namespaced results to a central LoggingManager. This is a move-only type.
 */
class PerformanceLogger {
public:
    // Rule of 5: Deleted copy, implemented move
    PerformanceLogger(const PerformanceLogger&) = delete;
    PerformanceLogger& operator=(const PerformanceLogger&) = delete;

    PerformanceLogger(PerformanceLogger&& other) noexcept;
    PerformanceLogger& operator=(PerformanceLogger&& other) noexcept;

    ~PerformanceLogger();

    /**
     * @brief Records the time elapsed since the last record in this scope.
     */
    void record(std::string_view name, cudaStream_t stream);

    /**
     * @brief Creates a nested performance logging scope.
     * @return A new PerformanceLogger for the nested scope.
     */
    PerformanceLogger scope(std::string_view name_prefix, cudaStream_t stream);

private:
    friend class LoggingManager; // Allow LoggingManager to construct this
    PerformanceLogger(LoggingManager* manager, std::string_view prefix, cudaStream_t stream);

    LoggingManager* manager_;
    std::string prefix_;
    cudaEvent_t last_event_;
};

/**
 * @class LoggingManager
 * @brief The top-level class that owns all timing data and acts as a factory
 * for PerformanceLogger scopes.
 */
class LoggingManager {
public:
    explicit LoggingManager(bool enabled = true);
    ~LoggingManager();

    // The LoggingManager is the top-level owner, so it should not be copied or moved.
    LoggingManager(const LoggingManager&) = delete;
    LoggingManager& operator=(const LoggingManager&) = delete;

    /**
     * @brief Creates a top-level performance logging scope.
     * @return A PerformanceLogger for the new scope.
     */
    PerformanceLogger scope(std::string_view name_prefix, cudaStream_t stream);

    /**
     * @brief Clears all recorded timing data.
     */
    void reset();

    /**
     * @brief Builds and prints a hierarchical tree report of all collected timings.
     */
    void print_report() const;

private:
    friend class PerformanceLogger; // Allow logger to add data
    void add_timing(std::string_view name, float elapsed_ms);

    // Private helper methods for printing the report tree
    static double calculate_total_times(ProfileNode& node);
    static void print_node_recursive(const ProfileNode& node, const std::string& name, const std::string& indent, bool is_last, double parent_total_ms);

    std::map<std::string, std::vector<float>> timings_;
    bool enabled_;
};

// --- Implementation of PerformanceLogger Methods ---

inline PerformanceLogger::PerformanceLogger(LoggingManager* manager, std::string_view prefix, cudaStream_t stream)
    : manager_(manager), prefix_(prefix), last_event_(nullptr) {
    // If the manager is null, this logger is disabled. Do not create any CUDA events.
    if (!manager_) {
        return;
    }

    if (!prefix.empty()) {
        prefix_ += ".";
    }
    CUDA_CHECK_PROFILER(cudaEventCreate(&last_event_));
    CUDA_CHECK_PROFILER(cudaEventRecord(last_event_, stream));
}

inline PerformanceLogger::PerformanceLogger(PerformanceLogger&& other) noexcept
    : manager_(other.manager_),
      prefix_(std::move(other.prefix_)),
      last_event_(other.last_event_) {
    other.last_event_ = nullptr;
}

inline PerformanceLogger& PerformanceLogger::operator=(PerformanceLogger&& other) noexcept {
    if (this != &other) {
        if (last_event_) cudaEventDestroy(last_event_);
        manager_ = other.manager_;
        prefix_ = std::move(other.prefix_);
        last_event_ = other.last_event_;
        other.last_event_ = nullptr;
    }
    return *this;
}

inline PerformanceLogger::~PerformanceLogger() {
    if (last_event_) {
        // Suppress errors on destruction as it can happen during stack unwinding
        cudaEventDestroy(last_event_);
    }
}

inline void PerformanceLogger::record(std::string_view name, cudaStream_t stream) {
    // If the logger is disabled (null manager) or the event is null, do nothing.
    if (!manager_ || !last_event_) return;

    cudaEvent_t current_event;
    CUDA_CHECK_PROFILER(cudaEventCreate(&current_event));
    CUDA_CHECK_PROFILER(cudaEventRecord(current_event, stream));

    float elapsed_ms;
    CUDA_CHECK_PROFILER(cudaEventSynchronize(current_event));
    CUDA_CHECK_PROFILER(cudaEventElapsedTime(&elapsed_ms, last_event_, current_event));

    manager_->add_timing(prefix_ + std::string(name), elapsed_ms);

    CUDA_CHECK_PROFILER(cudaEventDestroy(last_event_));
    last_event_ = current_event;
}

inline PerformanceLogger PerformanceLogger::scope(std::string_view name_prefix, cudaStream_t stream) {
    // Propagate the disabled state. If this logger is disabled, the new one will be too.
    if (!manager_) {
        return PerformanceLogger(nullptr, "", stream);
    }
    return PerformanceLogger(manager_, prefix_ + std::string(name_prefix), stream);
}

// --- Implementation of LoggingManager Methods ---

inline LoggingManager::LoggingManager(bool enabled) : enabled_(enabled) {}
inline LoggingManager::~LoggingManager() = default;

inline PerformanceLogger LoggingManager::scope(std::string_view name_prefix, cudaStream_t stream) {
    // If not enabled, return a disabled logger by passing a null manager.
    if (!enabled_) {
        return PerformanceLogger(nullptr, "", stream);
    }
    return PerformanceLogger(this, name_prefix, stream);
}

inline void LoggingManager::add_timing(std::string_view name, float elapsed_ms) {
    // A guard here is good practice, though PerformanceLogger's check should prevent this.
    if (!enabled_) return;
    timings_[std::string(name)].push_back(elapsed_ms);
}

inline void LoggingManager::reset() {
    if (!enabled_) return;
    timings_.clear();
}

inline void LoggingManager::print_report() const {
    if (!enabled_) return;
    
    ProfileNode root;

    for (const auto& [fullname, times] : timings_) {
        std::stringstream ss(fullname);
        std::string segment;
        ProfileNode* current_node = &root;

        while (std::getline(ss, segment, '.')) {
            current_node = &current_node->children[segment];
        }

        if (!times.empty()) {
            current_node->count = times.size();
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            current_node->mean_ms = sum / current_node->count;
            double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
            current_node->std_dev = (current_node->count > 1) ? std::sqrt(std::max(0.0, sq_sum / current_node->count - current_node->mean_ms * current_node->mean_ms)) : 0.0;
        }
    }

    calculate_total_times(root);

    std::cout << "\n--- 🌲 Performance Report 🌲 ---\n\n";
    std::cout << std::left << std::setw(50) << "Operation"
              << std::setw(20) << "Avg Latency (ms)"
              << std::setw(15) << "% of Parent"
              << std::setw(20) << "Std Dev (ms)"
              << std::setw(10) << "Samples" << "\n";
    std::cout << std::string(115, '-') << "\n";

    for (auto it = root.children.begin(); it != root.children.end(); ++it) {
        print_node_recursive(it->second, it->first, "", std::next(it) == root.children.end(), root.total_mean_ms);
    }
    std::cout << std::string(115, '-') << "\n";
}

inline double LoggingManager::calculate_total_times(ProfileNode& node) {
    double children_total = 0.0;
    for (auto& entry : node.children) {
        children_total += calculate_total_times(entry.second);
    }
    node.total_mean_ms = node.mean_ms + children_total;
    return node.total_mean_ms;
}

inline void LoggingManager::print_node_recursive(const ProfileNode& node, const std::string& name, const std::string& indent, bool is_last, double parent_total_ms) {
    std::cout << std::left << indent << (is_last ? "└── " : "├── ") << std::setw(46 - indent.length()) << name;
    std::cout << std::fixed << std::setprecision(4) << std::setw(20) << node.total_mean_ms;

    if (parent_total_ms > 1e-6) {
        double percent_of_parent = (node.total_mean_ms / parent_total_ms) * 100.0;
        std::string percent_str = std::to_string(percent_of_parent);
        std::cout << std::fixed << std::setprecision(1) << std::setw(14) << percent_str.substr(0, percent_str.find('.') + 2) + "%";
    } else {
        std::cout << std::setw(15) << " ";
    }

    if (node.count > 0) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(20) << node.std_dev
                  << std::setw(10) << node.count;
    }
    std::cout << "\n";

    std::string child_indent = indent + (is_last ? "    " : "│   ");
    for (auto it = node.children.begin(); it != node.children.end(); ++it) {
        print_node_recursive(it->second, it->first, child_indent, std::next(it) == node.children.end(), node.total_mean_ms);
    }
}

#endif // PROFILER_HPP