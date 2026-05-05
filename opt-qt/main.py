"""
Main Entry Point for OPT Model Quantization Pipeline.

This script provides a complete pipeline for:
1. Calibration: Collect quantization scales
2. Quantization: Apply quantization to the model
3. Evaluation: Evaluate perplexity on test dataset

Usage:
    python main.py --config config/Int8.yaml --model-path /path/to/opt-model
"""

import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant import load_config, QuantStatManager
from quant.model_wrapper import wrap_model_by_family
from quant.qwen_wrapper import switch_quantization_mode_all
from data import CalibrationDataLoader
from evaluation import evaluate_perplexity
from tqdm import tqdm
from quant.quant_linear import QuantizedLinear
from quant.quant_matmul import QuantizedMatMul

def validate_reuse_layers_have_scales(model):
    missing = []
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            action = m._resolve_calibration_action()
            if action == "reuse" and (not m._scale_files_exist()):
                missing.append(f"{m.layer_name}_{m.layer_idx}")
    if missing:
        raise FileNotFoundError(
            "These layers are set to reuse but scale files are missing:\n" +
            "\n".join(missing)
        )

def calibration_action_summary(model):
    summary = {"reuse": 0, "recalibrate": 0}
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            summary[m._resolve_calibration_action()] += 1
    return summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete OPT Model Quantization Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to quantization configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained OPT model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration if scales already exist"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation after quantization"
    )
    
    return parser.parse_args()


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def build_wrapped_model(args, config, scale_dir, mode="scale_inspection"):
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.get("model", {}).get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = config["model"]["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    stat_manager = QuantStatManager(scale_dir)
    model = wrap_model_by_family(
        model,
        config["quantization"],
        mode=mode,
        stat_manager=stat_manager,
    )

    return model, stat_manager


def calibrate(args, config):
    print_header("STEP 1: CALIBRATION")

    scale_dir = config["quantization"]["scale_dir"]
    os.makedirs(scale_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model, stat_manager = build_wrapped_model(
        args=args,
        config=config,
        scale_dir=scale_dir,
        mode="scale_inspection",
    )

    validate_reuse_layers_have_scales(model)
    summary = calibration_action_summary(model)
    print(f"Calibration action summary: {summary}")

    # 只有所有层都是 reuse，才允许真正跳过 calibration。
    if args.skip_calibration:
        if summary["recalibrate"] == 0:
            print("✓ All quantized layers are reuse; skipping calibration.")
            return model
        else:
            print(
                "⚠ --skip-calibration is set, but some layers still require recalibration. "
                "Continuing calibration."
            )

    if summary["recalibrate"] == 0:
        print("All quantized layers are reuse; skip calibration dataloader.")
        return model

    print("\nPreparing calibration data...")
    calib_cfg = config["calibration"]

    calib_loader = CalibrationDataLoader(
        dataset_name=calib_cfg["dataset"],
        dataset_config=calib_cfg.get("dataset_config", None),
        split=calib_cfg.get("split", "train"),
        model_name_or_path=args.model_path,
        seq_length=calib_cfg.get("seq_length", 2048),
        batch_size=calib_cfg.get("batch_size", 1),
        num_samples=calib_cfg.get("num_samples", None),
        seed=calib_cfg.get("seed", 42),
        text_column=calib_cfg.get("text_column", "text"),
        streaming=calib_cfg.get("streaming", False),
        min_text_tokens=calib_cfg.get("min_text_tokens", 32),
    )

    print(f"\nRunning calibration on {len(calib_loader)} batches...")
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(calib_loader, desc="Calibrating")):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    calibration_time = time.time() - start_time

    print("\n" + "-" * 80)
    stat_manager.print_summary()

    print("Saving quantization scales...")
    stat_manager.save_all_scales()

    print(f"\n✓ Calibration completed in {calibration_time:.2f}s")
    print(f"✓ Scales saved to: {scale_dir}")

    return model

def evaluate(args, config, model):
    print_header("STEP 2: QUANTIZATION & EVALUATION")

    stat_manager = QuantStatManager(config["quantization"]["scale_dir"])
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module._stat_manager = stat_manager

    model = switch_quantization_mode_all(model, "quant_forward")

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nApplying quantization (quant_forward mode)...")
    print("✓ Quantization applied successfully")

    if args.skip_evaluation:
        print("\n✓ Skipping evaluation (--skip-evaluation flag set)")
        return

    print("\n" + "-" * 80)
    print("STEP 3: PERPLEXITY EVALUATION")
    print("-" * 80)

    results = {}

    for eval_ds in config["evaluation"]["datasets"]:
        dataset_name = eval_ds["name"]
        dataset_cfg = eval_ds.get("config", None)
        split = eval_ds.get("split", "test")

        print(f"\nEvaluating on {dataset_name} ({dataset_cfg}, {split})...")

        start_time = time.time()

        ppl = evaluate_perplexity(
            model,
            tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_cfg,
            split=split,
            seq_length=eval_ds.get(
                "seq_length",
                config["evaluation"].get("seq_length", 2048),
            ),
            device=args.device,
            text_column=eval_ds.get("text_column", "text"),
            streaming=eval_ds.get("streaming", False),
            max_eval_tokens=eval_ds.get(
                "max_eval_tokens",
                config["evaluation"].get("max_eval_tokens", None),
            ),
            min_doc_tokens=eval_ds.get(
                "min_doc_tokens",
                config["evaluation"].get("min_doc_tokens", 128),
            ),
        )

        eval_time = time.time() - start_time

        eval_key = eval_ds.get(
            "alias",
            f"{dataset_name}:{dataset_cfg}:{split}",
        )

        results[eval_key] = {
            "perplexity": ppl,
            "time": eval_time,
        }

    print("\n" + "-" * 80)
    print("ZERO ACTIVATION RATIO")
    print("-" * 80)

    if stat_manager.total_element_count > 0:
        print(
            f"全模型量化激活零值比例: "
            f"{stat_manager.total_zero_count / stat_manager.total_element_count:.4%}"
        )
        print(
            f"零值元素: "
            f"{stat_manager.total_zero_count:,} / {stat_manager.total_element_count:,}"
        )

        if stat_manager.total_bit_count > 0:
            print(
                f"全模型稀疏比特比例: "
                f"{stat_manager.total_sparsebit_count / stat_manager.total_bit_count:.4%}"
            )
            print(
                f"全模型Sign Magnitude编码比例: "
                f"{stat_manager.total_amplitude_zero_bits_total / stat_manager.total_bit_count:.4%}"
            )
            print(
                f"零比特比例: "
                f"{stat_manager.total_0bit_count / stat_manager.total_bit_count:.4%}"
            )
    else:
        print("未收集到激活统计数据")

    print("\n" + "=" * 80)
    print("FINAL RESULTS".center(80))
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Perplexity: {result['perplexity']:.4f}")
        print(f"  Time: {result['time']:.2f}s")

    print("\n" + "=" * 80)

    del model
    torch.cuda.empty_cache()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Scale directory: {config['quantization']['scale_dir']}")
    
    try:
        # Step 1: Calibration
        model = calibrate(args, config)
        
        # Step 2 & 3: Quantization and Evaluation
        evaluate(args, config, model)
        
        print_header("✓ PIPELINE COMPLETED SUCCESSFULLY")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
