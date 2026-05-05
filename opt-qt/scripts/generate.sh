#!/bin/bash
# Text generation script for OPT model quantization

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate awq

# Configuration
CONFIG="config/Int8.yaml"
MODEL_PATH="/home/zyzhao/lfw_opt/models/opt-1.3b"
PROMPT="Once upon a time"

# Generate text
echo "Generating text with prompt: $PROMPT"
python run.py \
    --config $CONFIG \
    --model-path $MODEL_PATH \
    --mode generate \
    --device cuda \
    --prompt "$PROMPT"

echo "Generation complete!"

