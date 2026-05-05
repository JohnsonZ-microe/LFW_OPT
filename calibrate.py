"""
Calibration Script for OPT Model Quantization.

This script performs quantization calibration by collecting scale statistics
from a calibration dataset.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant import (
    load_config,
    QuantStatManager,
    wrap_model_by_family,
)
from data import CalibrationDataLoader
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calibrate OPT model for quantization")
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
        help="Device to use for calibration (default: cuda)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for scales (overrides config)"
    )
    
    return parser.parse_args()


def calibrate(args):
    """
    Perform quantization calibration.
    
    Args:
        args: Command line arguments
    """
    print("="*80)
    print("OPT Model Quantization Calibration")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['quantization']['scale_dir'] = args.output_dir
    
    scale_dir = config['quantization']['scale_dir']
    print(f"Scales will be saved to: {scale_dir}")
    
    # Create scale directory
    os.makedirs(scale_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"\nLoading model from: {args.model_path}")
    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.get("model", {}).get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = config["model"]["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create statistics manager
    stat_manager = QuantStatManager(scale_dir)
    
    # Wrap model with quantized layers in scale_inspection mode
    print("\nWrapping model with quantized layers (scale_inspection mode)...")
    model = wrap_model_by_family(
        model,
        config["quantization"],
        mode="scale_inspection",
        stat_manager=stat_manager,
    )
    
    # Prepare calibration data
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
    
    # Run calibration
    print("\nRunning calibration...")
    print(f"Processing {len(calib_loader)} batches...")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(calib_loader, desc="Calibrating")):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            
            # Forward pass (scales are collected automatically)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(calib_loader)} batches")
    
    # Print statistics summary
    print("\nCalibration complete!")
    stat_manager.print_summary()
    
    # Save scales
    print("\nSaving quantization scales...")
    stat_manager.save_all_scales()
    
    print("\n" + "="*80)
    print("Calibration finished successfully!")
    print(f"Scales saved to: {scale_dir}")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Run calibration
    try:
        calibrate(args)
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

