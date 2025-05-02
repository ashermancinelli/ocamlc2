#!/bin/bash
# Script to build and run code coverage for ocamlc2

set -e

# Directory where script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set build directory
BUILD_DIR="${PROJECT_ROOT}/build-coverage"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting code coverage build and analysis...${NC}"

# Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with coverage enabled
echo -e "${YELLOW}Configuring project with coverage enabled...${NC}"
cmake -G Ninja -DENABLE_COVERAGE=ON "${PROJECT_ROOT}"

# Build the project
echo -e "${YELLOW}Building project...${NC}"
ninja

# Run coverage target
echo -e "${YELLOW}Running tests with coverage...${NC}"
ninja coverage

# Check if coverage generation was successful
if [ -f "${BUILD_DIR}/coverage/coverage.lcov" ]; then
    # Show summary report
    echo -e "${GREEN}Coverage Summary:${NC}"
    cat coverage/summary.txt

    # Open HTML report if browser available
    if command -v open &> /dev/null; then
        echo -e "${GREEN}Opening coverage report in browser...${NC}"
        open coverage/html/index.html
    else
        echo -e "${GREEN}Coverage report generated at:${NC}"
        echo "open ${BUILD_DIR}/coverage/html/index.html"
    fi
    echo -e "\n${GREEN}Coverage analysis complete!${NC}"
else
    echo -e "${RED}Coverage generation failed - no coverage data was produced${NC}"
    echo -e "${YELLOW}Check the build logs for errors${NC}"
    exit 1
fi 
