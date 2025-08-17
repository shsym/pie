# Makefile for Metal kernel reference implementations (Linux testing)

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Build directory
BUILD_DIR = build

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Targets
all: $(BUILD_DIR)/test_metal_gemm $(BUILD_DIR)/test_metal_embedding $(BUILD_DIR)/test_metal_rmsnorm $(BUILD_DIR)/test_metal_silu_and_mul $(BUILD_DIR)/test_metal_extract_k_values

$(BUILD_DIR)/test_metal_gemm: src/metal_gemm_reference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/test_metal_embedding: src/metal_embedding_reference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/test_metal_rmsnorm: src/metal_rmsnorm_reference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/test_metal_silu_and_mul: src/metal_silu_and_mul_reference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/test_metal_extract_k_values: src/metal_extract_k_values_reference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Run tests against CUDA artifacts
test: $(BUILD_DIR)/test_metal_gemm $(BUILD_DIR)/test_metal_embedding $(BUILD_DIR)/test_metal_rmsnorm $(BUILD_DIR)/test_metal_silu_and_mul $(BUILD_DIR)/test_metal_extract_k_values
	@echo "=== Running Metal Phase 1A Tests Against CUDA Golden References ==="
	@echo ""
	@echo "Running Metal GEMM reference test..."
	$(BUILD_DIR)/test_metal_gemm
	@echo ""
	@echo "Running Metal Embedding reference test..."
	$(BUILD_DIR)/test_metal_embedding
	@echo ""
	@echo "Running Metal RMSNorm reference test..."
	$(BUILD_DIR)/test_metal_rmsnorm
	@echo ""
	@echo "Running Metal SiLU and Mul reference test..."
	$(BUILD_DIR)/test_metal_silu_and_mul
	@echo ""
	@echo "Running Metal Extract K Values reference test..."
	$(BUILD_DIR)/test_metal_extract_k_values
	@echo ""
	@echo "=== All Phase 1A Metal Tests Completed ==="

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test clean