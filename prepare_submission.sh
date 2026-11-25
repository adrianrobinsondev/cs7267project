#!/bin/bash

# Define the submission name
SUBMISSION_NAME="CS7267_FinalProject_Submission_$(date +%Y%m%d)"
ZIP_FILE="${SUBMISSION_NAME}.zip"

echo "Preparing generalizable framework submission zip: ${ZIP_FILE}..."

# Create a temporary directory for the zip content
mkdir -p submission_tmp/src
mkdir -p submission_tmp/data
mkdir -p submission_tmp/results

# Copy the source code
cp finalproject/src/*.py submission_tmp/src/

# Copy the top-level orchestrator script
cp finalproject/run_comparative_study.py submission_tmp/

# Copy data and results
if [ -d "finalproject/data" ]; then
    cp finalproject/data/*.csv submission_tmp/data/ 2>/dev/null || true
fi

if [ -d "finalproject/results" ]; then
    cp finalproject/results/*.csv submission_tmp/results/ 2>/dev/null || true
fi

# Zip the temporary directory
zip -r "${ZIP_FILE}" submission_tmp

# Cleanup
rm -rf submission_tmp

echo "--------------------------------------------------"
echo "Success! Your submission zip is ready: ${ZIP_FILE}"
echo "Contents:"
echo "- run_comparative_study.py (Top-level Orchestrator)"
echo "- src/ (Modular Framework)"
echo "- data/ (Processed Macro Data)"
echo "- results/ (Comparative Study CSV)"
echo "--------------------------------------------------"
