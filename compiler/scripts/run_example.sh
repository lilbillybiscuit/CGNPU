#!/bin/bash

set -e

cd "$(dirname "$0")/.."

echo "Building..."
scripts/build.sh

INPUT_NAME="${1:-example_matrix_input}"
INPUT_FILE="programs/test_inputs/${INPUT_NAME}.txt"

EXPECTED_OUTPUT_FILE="programs/test_outputs/${INPUT_NAME}_output.txt"

if [ ! -f "${INPUT_FILE}" ]; then
    echo "Input file ${INPUT_FILE} not found!"
    exit 1
fi

echo "Running matrix multiplication with input: ${INPUT_NAME}"
echo "==============================================="

echo -e "\n1. Compiling to bytecode (if needed):"
echo "-----------------------------------------"
if [ ! -f "$(pwd)/programs/matrix_mult/main.cpp.jsonl" ]; then
    build/compiler/compiler $(pwd)/programs/matrix_mult/main.cpp
    echo "Bytecode generated successfully."
else
    echo "Using existing bytecode."
fi
echo "-----------------------------------------"

echo -e "\n2. Running with heterogeneous runtime (${INPUT_NAME}):"
echo "-----------------------------------------"
echo "This will demonstrate work stealing across devices..."
echo "Processing large matrix (${INPUT_NAME})..."

TEMP_OUTPUT=$(mktemp)

cat "${INPUT_FILE}" | build/runtime/runtime $(pwd)/programs/matrix_mult/main.cpp.jsonl > "${TEMP_OUTPUT}" 2>&1

MATRIX_SIZE=$(head -n 1 "${INPUT_FILE}")
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"

grep -E "^([0-9]+ )+[0-9]+$" "${TEMP_OUTPUT}" | head -n ${MATRIX_SIZE} > "${TEMP_OUTPUT}.matrix"

if [ -f "${EXPECTED_OUTPUT_FILE}" ]; then
    echo "Validating output against expected result..."

    if diff -w "${TEMP_OUTPUT}.matrix" "${EXPECTED_OUTPUT_FILE}" > /dev/null; then
        echo " SUCCESS: Output matches expected result!"
    else
        echo " ERROR: Output does not match expected result!"
        echo "First few differences:"
        diff -w "${TEMP_OUTPUT}.matrix" "${EXPECTED_OUTPUT_FILE}" | head -n 5
    fi
else
    echo "No expected output file found at ${EXPECTED_OUTPUT_FILE}"
    echo "Can't validate results."
fi

echo ""
echo "Profiling Information:"

START_LINES=$(grep -n "=== HETEROGENEOUS EXECUTION PERFORMANCE SUMMARY ===" "${TEMP_OUTPUT}" | cut -d: -f1)

if [ ! -z "$START_LINES" ]; then
    FIRST_START=$(echo "$START_LINES" | head -n1)
    sed -n "${FIRST_START},/===================================/p" "${TEMP_OUTPUT}" | head -n$(grep -n "===================================" "${TEMP_OUTPUT}" | head -n1 | cut -d: -f1)
fi

echo ""
echo "Debug Information (partial):"
grep -E "Worker|thread|DEBUG|stealing|stalled|exited" "${TEMP_OUTPUT}" | head -n 10

rm "${TEMP_OUTPUT}" "${TEMP_OUTPUT}.matrix"