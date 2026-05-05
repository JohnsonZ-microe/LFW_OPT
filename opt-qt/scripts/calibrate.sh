#!/bin/bash
# Calibration script for OPT model quantization

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate awq

# Configuration
CONFIG="config/Int8.yaml"
MODEL_PATH="/home/zyzhao/lfw_opt/models/opt-1.3b"

# Run calibration
echo "Starting calibration with configuration: $CONFIG"
python calibrate.py \
    --config $CONFIG \
    --model-path $MODEL_PATH \
    --device cuda

echo "Calibration complete!"

