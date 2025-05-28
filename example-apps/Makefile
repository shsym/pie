
# Makefile for Symphony Example Apps
# This file contains build targets for the example applications

# Default target
.DEFAULT_GOAL := build

# Declare phony targets (targets that don't create files with the same name)
.PHONY: build clean help install

# Build the WebAssembly modules
build:
	cargo build --target wasm32-wasip2 --release

# Clean build artifacts
clean:
	cargo clean

# Install dependencies
install:
	rustup target add wasm32-wasip2

# Display help information
help:
	@echo "Available targets:"
	@echo "  build    - Build WebAssembly modules (default)"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install wasm32-wasip2 target"
	@echo "  help     - Show this help message"
