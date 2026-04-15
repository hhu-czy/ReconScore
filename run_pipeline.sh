#!/bin/bash

# ================= Configuration Area =================
MODEL_PATH="./model/Z-Image"
REF_DIR="/your/directory/to/original/images"
JSON_DIR="/your/directory/to/JSON/files"
GEN_DIR="/your/directory/to/synthetic/images"
DEVICE="cuda:0"
# ====================================================

echo "=========================================="
echo "Phase 1: Start batch generation of remote sensing images"
echo "=========================================="
python batch_generate.py \
    --json_dir "$JSON_DIR" \
    --ref_dir "$REF_DIR" \
    --out_dir "$GEN_DIR" \
    --model_path "$MODEL_PATH" \
    --max_side 1024 \
    --steps 28 \
    --device "$DEVICE"

echo -e "\n=========================================="
echo "Phase 2: Start batch evaluation of DreamSim similarity"
echo "=========================================="
# Note: --save_score flag is used to ensure the ReconScore is written back to the JSON files
python batch_evaluate.py \
    --json_dir "$JSON_DIR" \
    --ref_dir "$REF_DIR" \
    --gen_dir "$GEN_DIR" \
    --device "$DEVICE" \
    --save_score 

echo -e "\n[*] Entire pipeline execution completed successfully!"