#!/bin/bash

# Artifacts Transfer Script - Compress and Extract CUDA golden reference artifacts
# Usage: ./scripts/artifacts_transfer.sh [compress|extract]
# 
# This script enables seamless transfer of CUDA golden reference artifacts
# between Linux (generation) and macOS (Metal validation) machines.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_ARCHIVE="$WORKSPACE_ROOT/cuda_artifacts.tar.xz"
ARTIFACTS_SOURCE="$WORKSPACE_ROOT/metal-protocol-tests/tests/artifacts"
MANIFEST_FILE="$WORKSPACE_ROOT/metal-protocol-tests/tests/artifact_manifest.json"
MANIFEST_GENERATOR="$SCRIPT_DIR/generate_artifact_manifest.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [compress|extract]"
    echo ""
    echo "Commands:"
    echo "  compress  - Compress artifacts from Linux machine for transfer"
    echo "  extract   - Extract artifacts on macOS machine for Metal validation"
    echo ""
    echo "Examples:"
    echo "  # On Linux (after generating CUDA golden references)"
    echo "  ./scripts/artifacts_transfer.sh compress"
    echo ""
    echo "  # On macOS (before Metal testing)"
    echo "  ./scripts/artifacts_transfer.sh extract"
    echo ""
    echo "Archive location: cuda_artifacts.tar.xz (workspace root)"
    echo "Artifacts path:   metal-protocol-tests/tests/artifacts/"
}

check_requirements() {
    if ! command -v tar &> /dev/null; then
        echo -e "${RED}Error: tar command not found${NC}"
        exit 1
    fi
    
    if ! command -v xz &> /dev/null; then
        echo -e "${RED}Error: xz compression not available${NC}"
        echo "Install xz: apt-get install xz-utils (Linux) or brew install xz (macOS)"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found (required for artifact validation)${NC}"
        exit 1
    fi
}

validate_artifacts() {
    local should_exit_on_failure=${1:-true}
    
    echo -e "${BLUE}🔍 Validating artifact completeness...${NC}"
    
    if [ ! -f "$MANIFEST_GENERATOR" ]; then
        echo -e "${YELLOW}Warning: Manifest generator not found, skipping validation${NC}"
        return 0
    fi
    
    # Generate or update manifest if artifacts exist
    if [ -d "$ARTIFACTS_SOURCE" ]; then
        python3 "$MANIFEST_GENERATOR" generate "$ARTIFACTS_SOURCE" >/dev/null 2>&1
    fi
    
    # Validate if both artifacts and manifest exist
    if [ -d "$ARTIFACTS_SOURCE" ] && [ -f "$MANIFEST_FILE" ]; then
        if python3 "$MANIFEST_GENERATOR" validate "$ARTIFACTS_SOURCE" "$MANIFEST_FILE" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Artifact validation passed${NC}"
            return 0
        else
            echo -e "${RED}❌ Artifact validation failed${NC}"
            python3 "$MANIFEST_GENERATOR" validate "$ARTIFACTS_SOURCE" "$MANIFEST_FILE" 2>&1 | sed 's/^/  /'
            
            if [ "$should_exit_on_failure" = true ]; then
                echo -e "${RED}Fix artifacts before compressing${NC}"
                exit 1
            fi
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  Skipping validation (missing artifacts or manifest)${NC}"
        return 0
    fi
}

compress_artifacts() {
    echo -e "${BLUE}🗜️  Compressing CUDA artifacts for transfer...${NC}"
    
    if [ ! -d "$ARTIFACTS_SOURCE" ]; then
        echo -e "${RED}Error: Artifacts directory not found: $ARTIFACTS_SOURCE${NC}"
        echo "Generate CUDA artifacts first using:"
        echo "  cd metal-protocol-tests && bash scripts/regenerate_artifacts.sh"
        exit 1
    fi
    
    # Check if artifacts exist
    ARTIFACT_COUNT=$(find "$ARTIFACTS_SOURCE" -name "*.bin" -o -name "*.json" | wc -l)
    if [ "$ARTIFACT_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: No artifacts found in $ARTIFACTS_SOURCE${NC}"
        echo "Generate CUDA artifacts first using the test framework"
        exit 1
    fi
    
    echo -e "${YELLOW}Found $ARTIFACT_COUNT artifact files${NC}"
    
    # Validate artifacts before compression
    validate_artifacts true
    
    # Create archive with progress
    cd "$WORKSPACE_ROOT"
    echo -e "${BLUE}Creating compressed archive: cuda_artifacts.tar.xz${NC}"
    
    # Use uncompressed tar first, then compress separately for reliability
    echo -e "${YELLOW}Step 1/2: Creating uncompressed tar...${NC}"
    tar -cf cuda_artifacts.tar \
        --directory="$WORKSPACE_ROOT" \
        metal-protocol-tests/tests/artifacts/ \
        metal-protocol-tests/tests/artifact_manifest.json
    
    if [ ! -f cuda_artifacts.tar ]; then
        echo -e "${RED}Error: Failed to create tar archive${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Step 2/2: Compressing with xz (this may take a few minutes)...${NC}"
    xz -T 0 cuda_artifacts.tar  # Use all CPU cores for faster compression
    
    if [ ! -f cuda_artifacts.tar.xz ]; then
        echo -e "${RED}Error: Failed to compress archive${NC}"
        exit 1
    fi
    
    # Get archive info
    ARCHIVE_SIZE=$(du -h "$ARTIFACTS_ARCHIVE" | cut -f1)
    COMPRESSION_RATIO=$(echo "scale=1; $(stat -f%z "$ARTIFACTS_ARCHIVE" 2>/dev/null || stat -c%s "$ARTIFACTS_ARCHIVE") * 100 / $(du -sb "$ARTIFACTS_SOURCE" | cut -f1)" | bc 2>/dev/null || echo "N/A")
    
    echo -e "${GREEN}✅ Compression complete!${NC}"
    echo -e "📦 Archive: ${BLUE}$ARTIFACTS_ARCHIVE${NC}"
    echo -e "📏 Size: ${YELLOW}$ARCHIVE_SIZE${NC}"
    echo -e "🗜️  Compression: ${YELLOW}${COMPRESSION_RATIO}%${NC} of original"
    echo ""
    echo -e "${BLUE}Transfer this file to your macOS machine and run:${NC}"
    echo -e "${YELLOW}./scripts/artifacts_transfer.sh extract${NC}"
}

extract_artifacts() {
    echo -e "${BLUE}📂 Extracting CUDA artifacts for Metal validation...${NC}"
    
    if [ ! -f "$ARTIFACTS_ARCHIVE" ]; then
        echo -e "${RED}Error: Archive not found: $ARTIFACTS_ARCHIVE${NC}"
        echo "Copy the cuda_artifacts.tar.xz file to the workspace root first"
        exit 1
    fi
    
    # Check archive integrity
    if ! tar -tJf "$ARTIFACTS_ARCHIVE" >/dev/null 2>&1; then
        echo -e "${RED}Error: Archive appears corrupted or invalid${NC}"
        exit 1
    fi
    
    # Get archive info
    ARCHIVE_SIZE=$(du -h "$ARTIFACTS_ARCHIVE" | cut -f1)
    FILE_COUNT=$(tar -tJf "$ARTIFACTS_ARCHIVE" | wc -l)
    
    echo -e "${YELLOW}Archive: $ARCHIVE_SIZE ($FILE_COUNT files)${NC}"
    
    # Extract with progress
    cd "$WORKSPACE_ROOT"
    echo -e "${BLUE}Extracting to exact original paths...${NC}"
    
    tar -xJf "$ARTIFACTS_ARCHIVE" 2>/dev/null
    
    # Verify extraction and validate artifacts
    if [ -d "$ARTIFACTS_SOURCE" ]; then
        EXTRACTED_COUNT=$(find "$ARTIFACTS_SOURCE" -name "*.bin" -o -name "*.json" | wc -l)
        echo -e "${GREEN}✅ Extraction complete!${NC}"
        echo -e "📁 Location: ${BLUE}$ARTIFACTS_SOURCE${NC}"
        echo -e "📄 Files: ${YELLOW}$EXTRACTED_COUNT artifacts${NC}"
        
        # Validate extracted artifacts
        validate_artifacts false
        
        echo ""
        echo -e "${BLUE}Ready for Metal testing! Run:${NC}"
        echo -e "${YELLOW}cd metal-protocol-tests/build${NC}"
        echo -e "${YELLOW}./metal_protocol_tests --backend metal --op gemm --case test1 --m 32 --n 128 --k 64${NC}"
    else
        echo -e "${RED}Error: Extraction failed - artifacts directory not created${NC}"
        exit 1
    fi
}

show_status() {
    echo -e "${BLUE}📊 Artifacts Transfer Status${NC}"
    echo ""
    
    if [ -f "$ARTIFACTS_ARCHIVE" ]; then
        ARCHIVE_SIZE=$(du -h "$ARTIFACTS_ARCHIVE" | cut -f1)
        echo -e "📦 Archive: ${GREEN}Found${NC} (${YELLOW}$ARCHIVE_SIZE${NC})"
    else
        echo -e "📦 Archive: ${RED}Not found${NC}"
    fi
    
    if [ -d "$ARTIFACTS_SOURCE" ]; then
        ARTIFACT_COUNT=$(find "$ARTIFACTS_SOURCE" -name "*.bin" -o -name "*.json" | wc -l)
        echo -e "📁 Artifacts: ${GREEN}Found${NC} (${YELLOW}$ARTIFACT_COUNT files${NC})"
        
        # Show validation status
        if [ -f "$MANIFEST_FILE" ]; then
            echo -e "📋 Manifest: ${GREEN}Found${NC}"
            validate_artifacts false >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "✅ Validation: ${GREEN}Passed${NC}"
            else
                echo -e "❌ Validation: ${RED}Failed${NC}"
            fi
        else
            echo -e "📋 Manifest: ${YELLOW}Missing${NC}"
        fi
        
        # Show operation coverage using manifest if available
        if [ -f "$MANIFEST_FILE" ] && [ -f "$MANIFEST_GENERATOR" ]; then
            echo -e "${BLUE}Operation Coverage:${NC}"
            python3 "$MANIFEST_GENERATOR" summary "$MANIFEST_FILE" 2>/dev/null | grep "  •" | sed 's/^//'
        else
            # Fallback to directory listing
            echo -e "${BLUE}Operation Coverage:${NC}"
            for op_dir in "$ARTIFACTS_SOURCE"/*; do
                if [ -d "$op_dir" ]; then
                    op_name=$(basename "$op_dir")
                    case_count=$(find "$op_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
                    echo -e "  • ${op_name}: ${YELLOW}${case_count} cases${NC}"
                fi
            done
        fi
    else
        echo -e "📁 Artifacts: ${RED}Not found${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    if [ ! -f "$ARTIFACTS_ARCHIVE" ] && [ ! -d "$ARTIFACTS_SOURCE" ]; then
        echo -e "  1. Generate CUDA artifacts: ${YELLOW}cd metal-protocol-tests && bash scripts/regenerate_artifacts.sh${NC}"
        echo -e "  2. Compress for transfer: ${YELLOW}./scripts/artifacts_transfer.sh compress${NC}"
    elif [ -d "$ARTIFACTS_SOURCE" ] && [ ! -f "$ARTIFACTS_ARCHIVE" ]; then
        echo -e "  • Compress artifacts: ${YELLOW}./scripts/artifacts_transfer.sh compress${NC}"
    elif [ -f "$ARTIFACTS_ARCHIVE" ] && [ ! -d "$ARTIFACTS_SOURCE" ]; then
        echo -e "  • Extract artifacts: ${YELLOW}./scripts/artifacts_transfer.sh extract${NC}"
    else
        echo -e "  • All ready! Test Metal: ${YELLOW}cd metal-protocol-tests/build && ./metal_protocol_tests --backend metal --op gemm --case test1 --m 32 --n 128 --k 64${NC}"
    fi
}

main() {
    check_requirements
    
    case "${1:-status}" in
        compress)
            compress_artifacts
            ;;
        extract)
            extract_artifacts
            ;;
        status)
            show_status
            ;;
        -h|--help|help)
            print_usage
            ;;
        *)
            echo -e "${RED}Error: Unknown command '$1'${NC}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

main "$@"