#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <sstream>
#include <algorithm>
#include <tuple>
#include <map>
#include <set>

// Forward declarations
class Profiler;
class ProfileScope;

// Helper macro to safely check CUDA calls
#define CUDA_CHECK_PROFILER(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        char error_buf[256]; \
        snprintf(error_buf, sizeof(error_buf), "CUDA Error in Profiler at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(error_buf); \
    } \
} while (0)

// A node in the performance report tree.
struct ProfileNode {
    double exclusive_sum_ms = 0.0;
    double exclusive_sum_sq_ms = 0.0;
    size_t count = 0;
    double inclusive_sum_ms = 0.0;

    // Stores children in insertion order. The string is the key/name.
    std::vector<std::pair<std::string, ProfileNode>> children;
    
    // For fast lookup of a child's index in the `children` vector.
    std::map<std::string, size_t> name_to_child_index;
};

class Profiler {
public:
    explicit Profiler(bool enabled = true);
    ~Profiler();

    // Non-copyable
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Creates a top-level profiling scope.
    ProfileScope scope(std::string_view name, cudaStream_t stream);
    void print_report();
    void reset();

private:
    friend class ProfileScope;
    
    cudaEvent_t create_event();
    void add_event_pair(const std::string& name, cudaEvent_t start, cudaEvent_t end, cudaStream_t stream);
    void process_events();

    static double calculate_inclusive_times(ProfileNode& node);
    static void print_node_recursive(const ProfileNode& node, const std::string& name, const std::string& indent, bool is_last, double parent_inclusive_ms, double total_root_ms, int depth);
    
    // Data is stored here to preserve order before being processed into the tree
    std::vector<std::tuple<std::string, cudaEvent_t, cudaEvent_t>> event_pairs_;
    std::vector<std::pair<std::string, float>> timed_events_;

    std::vector<cudaEvent_t> all_events_;
    std::set<cudaStream_t> streams_used_;
    bool enabled_;
};

class ProfileScope {
public:
    // RAII constructor for nested scopes.
    ProfileScope(ProfileScope& parent, std::string_view name);

    // Non-copyable, but movable.
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ProfileScope(ProfileScope&& other) noexcept;
    ProfileScope& operator=(ProfileScope&& other) noexcept;

    // Destructor updates parent's event timeline.
    ~ProfileScope();

    // Records a timed event within the current scope.
    void record(std::string_view name);

    // Creates and returns a temporary nested scope object.
    ProfileScope scope(std::string_view name) {
        return ProfileScope(*this, name);
    }

private:
    friend class Profiler; // Profiler can create root scopes.

    // Private constructor for top-level scopes.
    ProfileScope(Profiler* profiler, std::string name, cudaStream_t stream);

    Profiler* profiler_;
    ProfileScope* parent_;
    std::string name_prefix_;
    cudaStream_t stream_;
    cudaEvent_t last_event_;
    bool moved_from_ = false;
};

// --- Implementation of ProfileScope Methods ---

inline ProfileScope::ProfileScope(Profiler* profiler, std::string name, cudaStream_t stream)
    : profiler_(profiler), parent_(nullptr), name_prefix_(std::move(name)), stream_(stream), last_event_(nullptr) {
    if (profiler_ && profiler_->enabled_) {
        if (!name_prefix_.empty()) {
            name_prefix_ += ".";
        }
        last_event_ = profiler_->create_event();
        CUDA_CHECK_PROFILER(cudaEventRecord(last_event_, stream_));
    }
}

inline ProfileScope::ProfileScope(ProfileScope& parent, std::string_view name)
    : profiler_(parent.profiler_), parent_(&parent), stream_(parent.stream_), last_event_(parent.last_event_), moved_from_(false) {
    if (profiler_ && profiler_->enabled_) {
        name_prefix_ = parent.name_prefix_ + std::string(name) + ".";
    }
}

inline ProfileScope::ProfileScope(ProfileScope&& other) noexcept
    : profiler_(other.profiler_),
      parent_(other.parent_),
      name_prefix_(std::move(other.name_prefix_)),
      stream_(other.stream_),
      last_event_(other.last_event_),
      moved_from_(false) {
    other.moved_from_ = true; // Invalidate the other scope.
}

inline ProfileScope& ProfileScope::operator=(ProfileScope&& other) noexcept {
    if (this != &other) {
        profiler_ = other.profiler_;
        parent_ = other.parent_;
        name_prefix_ = std::move(other.name_prefix_);
        stream_ = other.stream_;
        last_event_ = other.last_event_;
        moved_from_ = false;
        other.moved_from_ = true;
    }
    return *this;
}

inline ProfileScope::~ProfileScope() {
    if (moved_from_ || !profiler_ || !profiler_->enabled_) return;

    if (parent_) {
        parent_->last_event_ = this->last_event_;
    }
}

inline void ProfileScope::record(std::string_view name) {
    if (!profiler_ || !profiler_->enabled_ || !last_event_) return;

    cudaEvent_t current_event = profiler_->create_event();
    CUDA_CHECK_PROFILER(cudaEventRecord(current_event, stream_));

    profiler_->add_event_pair(name_prefix_ + std::string(name), last_event_, current_event, stream_);
    last_event_ = current_event;
}


// --- Implementation of Profiler Methods ---

inline Profiler::Profiler(bool enabled) : enabled_(enabled) {}

inline Profiler::~Profiler() {
    for (cudaEvent_t event : all_events_) {
        cudaEventDestroy(event);
    }
}

inline ProfileScope Profiler::scope(std::string_view name, cudaStream_t stream) {
    return ProfileScope(this, std::string(name), stream);
}

inline cudaEvent_t Profiler::create_event() {
    if (!enabled_) return nullptr;
    cudaEvent_t event;
    CUDA_CHECK_PROFILER(cudaEventCreate(&event));
    all_events_.push_back(event);
    return event;
}

inline void Profiler::add_event_pair(const std::string& name, cudaEvent_t start, cudaEvent_t end, cudaStream_t stream) {
    if (!enabled_) return;
    event_pairs_.emplace_back(name, start, end);
    streams_used_.insert(stream);
}

inline void Profiler::reset() {
    if (!enabled_) return;
    for (cudaEvent_t event : all_events_) {
        cudaEventDestroy(event);
    }
    all_events_.clear();
    event_pairs_.clear();
    timed_events_.clear();
    streams_used_.clear();
}

inline void Profiler::process_events() {
    if (!enabled_ || !timed_events_.empty()) return; // Already processed

    for (cudaStream_t stream : streams_used_) {
        CUDA_CHECK_PROFILER(cudaStreamSynchronize(stream));
    }

    timed_events_.reserve(event_pairs_.size());
    for (const auto& [name, start_event, end_event] : event_pairs_) {
        float elapsed_ms;
        CUDA_CHECK_PROFILER(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
        timed_events_.emplace_back(name, elapsed_ms);
    }
    
    event_pairs_.clear();
    streams_used_.clear();
}

inline void Profiler::print_report() {
    if (!enabled_) return;
    
    process_events(); // Populates `timed_events_` in order.

    ProfileNode root;
    for (const auto& [fullname, time_ms] : timed_events_) {
        std::stringstream ss(fullname);
        std::string segment;
        ProfileNode* current_node = &root;

        while (std::getline(ss, segment, '.')) {
            if (!segment.empty()) {
                auto it = current_node->name_to_child_index.find(segment);
                if (it == current_node->name_to_child_index.end()) {
                    // Child not found, create it to preserve order.
                    current_node->children.emplace_back(segment, ProfileNode{});
                    size_t new_index = current_node->children.size() - 1;
                    current_node->name_to_child_index[segment] = new_index;
                    current_node = &current_node->children.back().second;
                } else {
                    // Child found, move to it.
                    current_node = &current_node->children[it->second].second;
                }
            }
        }

        // Aggregate timing data on the final leaf node.
        current_node->count++;
        current_node->exclusive_sum_ms += time_ms;
        current_node->exclusive_sum_sq_ms += time_ms * time_ms;
    }

    calculate_inclusive_times(root);
    const double total_root_ms = root.inclusive_sum_ms;

    std::cout << "\n--- 🌲 Performance Report (Execution Order) 🌲 ---\n\n";
    std::cout << std::left << std::setw(50) << "Operation"
              << std::right 
              << std::setw(20) << "Total Latency (ms)"
              << std::setw(15) << "% of Entire"
              << std::setw(15) << "% of Parent"
              << std::setw(20) << "Std Dev (ms)"
              << std::setw(10) << "Samples" << "\n";
    std::cout << std::string(130, '-') << "\n";

    // Iterate through the ordered vector of children to print the report.
    for (size_t i = 0; i < root.children.size(); ++i) {
        const auto& child_entry = root.children[i];
        bool is_last = (i == root.children.size() - 1);
        print_node_recursive(child_entry.second, child_entry.first, "", is_last, total_root_ms, total_root_ms, 0);
    }
    std::cout << std::string(130, '-') << "\n";
}

inline double Profiler::calculate_inclusive_times(ProfileNode& node) {
    double children_inclusive_sum = 0.0;
    // Iterate over the ordered vector.
    for (auto& child_pair : node.children) {
        children_inclusive_sum += calculate_inclusive_times(child_pair.second);
    }
    node.inclusive_sum_ms = node.exclusive_sum_ms + children_inclusive_sum;
    return node.inclusive_sum_ms;
}

inline void Profiler::print_node_recursive(const ProfileNode& node, const std::string& name, const std::string& indent, bool is_last, double parent_inclusive_ms, double total_root_ms, int depth) {
    std::string prefix = indent + (is_last ? "└── " : "├── ");
    size_t prefix_display_width = (depth * 4) + 4;
    std::cout << std::left << prefix;
    if (50 > prefix_display_width) {
        std::cout << std::setw(50 - prefix_display_width) << name;
    } else {
        std::cout << name;
    }

    std::cout << std::right << std::fixed << std::setprecision(4) << std::setw(20) << node.inclusive_sum_ms;
    
    if (total_root_ms > 1e-6) {
        double percent_of_entire = (node.inclusive_sum_ms / total_root_ms) * 100.0;
        std::stringstream percent_ss;
        percent_ss << std::fixed << std::setprecision(1) << percent_of_entire << "%";
        std::cout << std::right << std::setw(15) << percent_ss.str();
    } else {
        std::cout << std::right << std::setw(15) << " ";
    }
    
    if (parent_inclusive_ms > 1e-6 && node.inclusive_sum_ms > 0) {
        double percent_of_parent = (node.inclusive_sum_ms / parent_inclusive_ms) * 100.0;
        std::stringstream percent_ss;
        percent_ss << std::fixed << std::setprecision(1) << percent_of_parent << "%";
        std::cout << std::right << std::setw(15) << percent_ss.str();
    } else {
        std::cout << std::right << std::setw(15) << " ";
    }

    if (node.count > 0) {
        double mean = node.exclusive_sum_ms / node.count;
        double std_dev = (node.count > 1) ? std::sqrt(std::max(0.0, node.exclusive_sum_sq_ms / node.count - mean * mean)) : 0.0;
        std::cout << std::right << std::fixed << std::setprecision(4) << std::setw(20) << std_dev;
        std::cout << std::right << std::setw(10) << node.count;
    }
    std::cout << "\n";

    std::string child_indent = indent + (is_last ? "    " : "│   ");
    // Iterate over the ordered vector for recursive printing.
    for (size_t i = 0; i < node.children.size(); ++i) {
        const auto& child_entry = node.children[i];
        bool is_child_last = (i == node.children.size() - 1);
        print_node_recursive(child_entry.second, child_entry.first, child_indent, is_child_last, node.inclusive_sum_ms, total_root_ms, depth + 1);
    }
}
#endif // PROFILER_HPP