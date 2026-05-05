#!/bin/bash

# Configuration files to evaluate
CONFIGS=(
    # OPT configs
    "opt-qt/config/opt_1.3b_linear_matmul_int8.yaml"
    "opt-qt/config/opt_1.3b_linear_matmul_int12.yaml"
    "opt-qt/config/opt_1.3b_linear_matmul_int16.yaml"
    "opt-qt/config/opt_6.7b_linear_matmul_int8.yaml"
    "opt-qt/config/opt_6.7b_linear_matmul_int12.yaml"
    "opt-qt/config/opt_6.7b_linear_matmul_int16.yaml"
    "opt-qt/config/opt_13b_linear_matmul_int8.yaml"
    "opt-qt/config/opt_13b_linear_matmul_int12.yaml"
    "opt-qt/config/opt_13b_linear_matmul_int16.yaml"
    
    # Qwen configs
    "opt-qt/config/qwen2_1.5b_linear_matmul_int8.yaml"
    "opt-qt/config/qwen2_1.5b_linear_matmul_int12.yaml"
    "opt-qt/config/qwen2_1.5b_linear_matmul_int16.yaml"
    "opt-qt/config/qwen2_7b_linear_matmul_int8.yaml"
    "opt-qt/config/qwen2_7b_linear_matmul_int12.yaml"
    "opt-qt/config/qwen2_7b_linear_matmul_int16.yaml"
    "opt-qt/config/qwen2_14b_linear_matmul_int8.yaml"
    "opt-qt/config/qwen2_14b_linear_matmul_int12.yaml"
    "opt-qt/config/qwen2_14b_linear_matmul_int16.yaml"
)

OUT_FILE="experiment_results.csv"
echo "Config,ActivationZeroRatio,ZeroElements,Sparsity,SignMagRatio,ZeroBitRatio,Perplexity" > $OUT_FILE

echo "Starting evaluation loop..."

for config in "${CONFIGS[@]}"; do
    echo "--------------------------------------------------------"
    echo "Running with config: $config"
    
    # Use a temporary log file for each run in case we want to inspect it later
    TMP_LOG="current_run.log"
    
    # Extract model path from config
    model_path=$(grep -m 1 "^  path:" "$config" | awk '{print $2}')
    
    # Run the main.py script
    python opt-qt/main.py --config "$config" --model-path "$model_path" > "$TMP_LOG" 2>&1
    
    # Extract perplexity ("Perplexity: X.XXXX")
    ppl=$(grep "Perplexity:" "$TMP_LOG" | tail -n 1 | awk '{print $2}')
    if [ -z "$ppl" ]; then
        ppl="N/A"
    fi
    
    # Extract activation zero ratio
    act_zero=$(grep "全模型量化激活零值比例:" "$TMP_LOG" | awk -F': ' '{print $2}' | tr -d ' ')
    if [ -z "$act_zero" ]; then act_zero="N/A"; fi

    # Extract zero elements
    zero_elems=$(grep "零值元素:" "$TMP_LOG" | awk -F': ' '{print $2}' | tr -d ' ')
    if [ -z "$zero_elems" ]; then zero_elems="N/A"; fi

    # Extract sparsity ("全模型稀疏比特比例: X.XXXX%")
    sparsity=$(grep "全模型稀疏比特比例:" "$TMP_LOG" | awk -F': ' '{print $2}' | tr -d ' ')
    if [ -z "$sparsity" ]; then sparsity="N/A"; fi

    # Extract Sign Magnitude ratio
    sign_mag=$(grep "全模型Sign Magnitude编码比例:" "$TMP_LOG" | awk -F': ' '{print $2}' | tr -d ' ')
    if [ -z "$sign_mag" ]; then sign_mag="N/A"; fi

    # Extract zero bits
    zero_bits=$(grep "零比特比例:" "$TMP_LOG" | awk -F': ' '{print $2}' | tr -d ' ')
    if [ -z "$zero_bits" ]; then zero_bits="N/A"; fi

    # Save and display result
    echo "$config,$act_zero,$zero_elems,$sparsity,$sign_mag,$zero_bits,$ppl" >> "$OUT_FILE"
    echo "Finished $config -> ActZero: $act_zero | ZeroElems: $zero_elems | Sparsity: $sparsity | SignMag: $sign_mag | ZeroBits: $zero_bits | PPL: $ppl"
done

echo "--------------------------------------------------------"
echo "All done! Results saved to $OUT_FILE"
cat $OUT_FILE
