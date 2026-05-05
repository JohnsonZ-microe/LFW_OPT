import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.longbench_eval import run_longbench_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-implementation", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

    run_longbench_predictions(
        model=model,
        tokenizer=tokenizer,
        datasets=["qasper", "hotpotqa", "gov_report", "passage_retrieval_en"],
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_input_tokens=args.max_input_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()