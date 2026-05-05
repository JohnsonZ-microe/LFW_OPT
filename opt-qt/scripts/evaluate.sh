#!/bin/bash
# Evaluation script for OPT model quantization

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate awq

# Configuration
CONFIG="config/Int8.yaml"
MODEL_PATH="/home/zyzhao/lfw_opt/models/opt-1.3b"
OUTPUT_DIR="./results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Evaluate quantized model
echo "Evaluating quantized model..."
python run.py \
    --config $CONFIG \
    --model-path $MODEL_PATH \
    --mode evaluate \
    --device cuda \
    --output $OUTPUT_DIR/quantized_results.json

# Evaluate original model (for comparison)
echo "Evaluating original model..."
python run.py \
    --config $CONFIG \
    --model-path $MODEL_PATH \
    --mode evaluate \
    --device cuda \
    --no-quantization \
    --output $OUTPUT_DIR/original_results.json

echo "Evaluation complete! Results saved to $OUTPUT_DIR"

