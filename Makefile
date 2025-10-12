# Root Makefile for llm-cpp project
# Wraps CMake build system with convenient targets

BUILD_DIR = build
CMAKE_FLAGS =

# Default target
.PHONY: all
all: release

# Release build
.PHONY: release
release:
	@echo "Building llm-cpp (Release)..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo "Release build completed: $(BUILD_DIR)/bin/llm-cpp"

# Debug build with AddressSanitizer
.PHONY: debug
debug:
	@echo "Building llm-cpp (Debug with AddressSanitizer)..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo "Debug build completed: $(BUILD_DIR)/bin/llm-cpp"

# Clean build directory
.PHONY: clean
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean completed."

# Clean and rebuild release
.PHONY: rebuild
rebuild: clean release

# Clean and rebuild debug
.PHONY: rebuild-debug
rebuild-debug: clean debug

# Run the release executable
.PHONY: run
run: release
	@./$(BUILD_DIR)/bin/llm-cpp

# Run the debug executable
.PHONY: run-debug
run-debug: debug
	@./$(BUILD_DIR)/bin/llm-cpp

# Format code (requires clang-format)
.PHONY: format
format:
	@if [ -d "$(BUILD_DIR)" ]; then \
		cd $(BUILD_DIR) && make format; \
	else \
		echo "Build directory not found. Run 'make' first."; \
	fi

# Check code formatting
.PHONY: format-check
format-check:
	@if [ -d "$(BUILD_DIR)" ]; then \
		cd $(BUILD_DIR) && make format-check; \
	else \
		echo "Build directory not found. Run 'make' first."; \
	fi

# Lint code with clang-tidy
.PHONY: tidy
tidy:
	@if [ -d "$(BUILD_DIR)" ]; then \
		cd $(BUILD_DIR) && make tidy; \
	else \
		echo "Build directory not found. Run 'make' first."; \
	fi

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make, make all      - Build release version"
	@echo "  make release        - Build release version"
	@echo "  make debug          - Build debug version with AddressSanitizer"
	@echo "  make clean          - Clean build directory"
	@echo "  make rebuild        - Clean and rebuild release"
	@echo "  make rebuild-debug  - Clean and rebuild debug"
	@echo "  make run            - Build and run release executable"
	@echo "  make run-debug      - Build and run debug executable"
	@echo "  make format         - Format source code"
	@echo "  make format-check   - Check code formatting"
	@echo "  make tidy           - Run clang-tidy linter"
	@echo "  make help           - Show this help"
