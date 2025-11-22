#!/bin/bash

# Test script for Chapter 1 of Part B Instructions
# This script tests all non-interactive parts of Chapter 1 (sections 1.1, 1.2, and 1.3)
# Sections 1.4 and 1.5 are skipped (optional exercises)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Function to print step headers
print_step() {
    echo ""
    echo -e "${GREEN}>>> $1${NC}"
    echo ""
}

# Function to check if command succeeded
check_result() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        exit 1
    fi
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_section "Chapter 1 Test Suite - LLM Fine-Tuning Demo"

# Check if we're in the right directory
if [ ! -f "run_model.py" ] || [ ! -f "finetune_llm.py" ]; then
    echo -e "${RED}Error: This script must be run from the AI_With_Python_PartB directory${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not found. Please run Chapter 0 setup first.${NC}"
    exit 1
fi

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_step "Activating virtual environment"
    source venv/bin/activate
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Virtual environment activated${NC}"
    else
        echo -e "${RED}Failed to activate virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Virtual environment already activated${NC}"
fi

# ============================================
# Section 1.1: Load an LLM
# ============================================
print_section "Section 1.1: Load an LLM - Test Base Model"

print_step "1.1 Step 1: Testing base model (--use-raw) with Shakespeare prompts"
python run_model.py --use-raw --prompts-file data/shakespeare_prompts.json
check_result

# Note: Interactive mode (1.1 Step 2) is skipped as it requires user input

# ============================================
# Section 1.2: Fine Tune Your First LLM
# ============================================
print_section "Section 1.2: Fine Tune Your First LLM"

print_step "1.2 Step 1: Fine-tuning with washing machine data"
python finetune_llm.py --data data/washingmachine_data.txt
check_result

# Check if model was created
if [ -d "./fine_tuned_washingmachine_data_model" ]; then
    echo -e "${GREEN}✓ Fine-tuned model directory created${NC}"
else
    echo -e "${YELLOW}Warning: Model directory not found, but training may have completed${NC}"
fi

# ============================================
# Section 1.3: Run Fine Tuned LLM
# ============================================
print_section "Section 1.3: Run Fine Tuned LLM"

print_step "1.3 Step 1: Testing fine-tuned washing machine model with Shakespeare prompts"
python run_model.py --model-path ./fine_tuned_washingmachine_data_model --prompts-file data/shakespeare_prompts.json
check_result

print_step "1.3 Step 2: Testing fine-tuned washing machine model with washing machine prompts"
python run_model.py --model-path ./fine_tuned_washingmachine_data_model --prompts-file data/washingmachine_prompts.json
check_result

# Note: Interactive mode (1.3 Step 3) is skipped as it requires user input

# Note: Sections 1.4 and 1.5 are skipped (optional exercises for students to try on their own)

# ============================================
# Summary
# ============================================
print_section "Test Summary"

echo -e "${GREEN}All non-interactive tests completed successfully!${NC}"
echo ""
echo "Note: Interactive mode tests were skipped as they require user input."
echo "To test interactive mode manually, run:"
echo "  python run_model.py --use-raw --interactive"
echo "  python run_model.py --model-path ./fine_tuned_washingmachine_data_model --interactive"
echo ""

