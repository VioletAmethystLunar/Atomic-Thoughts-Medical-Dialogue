#!/bin/bash

# ==============================================================================
# Description:  the process of annotation, filtering, sampling, generation, and refinement.
# ==============================================================================

set -e

BASE_DIR=$(pwd)
ANNOTATE_DIR="$BASE_DIR/Annotate"

export DASHSCOPE_API_KEY="sk-xxx"  # add your API key

if [ "$DASHSCOPE_API_KEY" == "sk-xxx" ] || [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: Please set a valid DASHSCOPE_API_KEY in run_pipeline.sh"
    exit 1
fi

echo "Starting the Medical Dialogue Improvement Pipeline..."
echo "Working Directory: $ANNOTATE_DIR"

cd "$ANNOTATE_DIR"

# ------------------------------------------------------------------------------
# Step 1: Annotation (Labeling Intents and Atomic Actions)
# ------------------------------------------------------------------------------
echo -e "\n[1/6] Running Annotation (Intent & Atomic Action Labeling)..."
echo "      Script: annotate.py"
python annotate.py
echo "Annotation completed."

# ------------------------------------------------------------------------------
# Step 2: Filtering (Cleaning Data based on rules)
# ------------------------------------------------------------------------------
echo -e "\n[2/6] Running Data Filtering..."
echo "      Script: filter.py"
python filter.py
echo "Filtering completed."

# ------------------------------------------------------------------------------
# Step 3: Sampling (Data Balancing)
# ------------------------------------------------------------------------------
echo -e "\n[3/6] Running Data Sampling..."
echo "      Script: sample.py"
python sample.py
echo "Sampling completed."

# ------------------------------------------------------------------------------
# Step 4: Structure Matching (Intersecting Sampled & Filtered Data)
# ------------------------------------------------------------------------------
echo -e "\n[4/6] Running Structure Matching..."
echo "      Script: filtered_by_struct.py"
python filtered_by_struct.py
echo "Data matching completed."

# ------------------------------------------------------------------------------
# Step 5: Generation (Generating Chain-of-Thought)
# ------------------------------------------------------------------------------
echo -e "\n[5/6] Generating Thought Chains (CoT)..."
echo "      Script: generate.py"
python generate.py
echo "CoT generation completed."

# ------------------------------------------------------------------------------
# Step 6: Refinement (Polishing Doctor Responses)
# ------------------------------------------------------------------------------
echo -e "\n[6/6] Refining Doctor Responses..."
echo "      Script: refine.py"
python refine.py
echo "Refinement completed."

# ------------------------------------------------------------------------------
# Finish
# ------------------------------------------------------------------------------
echo -e "\nAll results are in the 'results' directory."