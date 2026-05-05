import os
import json
import torch
from datasets import load_dataset
from tqdm import tqdm


MAX_NEW_TOKENS = {
    "qasper": 128,
    "hotpotqa": 128,
    "gov_report": 512,
    "passage_retrieval_en": 32,
}


def build_longbench_prompt(example, dataset_name):
    context = example["context"]
    inp = example["input"]

    # 第一版 smoke prompt。正式报告建议换成 LongBench 官方 config 里的 prompt。
    return (
        f"Context:\n{context}\n\n"
        f"Instruction:\n{inp}\n\n"
        f"Answer:"
    )


@torch.no_grad()
def run_longbench_predictions(
    model,
    tokenizer,
    datasets,
    output_dir,
    max_samples=20,
    max_input_tokens=8192,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for ds_name in datasets:
        print(f"\nRunning LongBench subset: {ds_name}")
        ds = load_dataset("THUDM/LongBench", ds_name, split="test", trust_remote_code=True)

        out_path = os.path.join(output_dir, f"{ds_name}.jsonl")
        max_new = MAX_NEW_TOKENS.get(ds_name, 128)

        with open(out_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(tqdm(ds, desc=ds_name)):
                if max_samples is not None and i >= max_samples:
                    break

                prompt = build_longbench_prompt(ex, ds_name)

                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                )

                encoded = {k: v.to(device) for k, v in encoded.items()}

                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                gen_ids = output_ids[0, encoded["input_ids"].shape[1]:]
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

                rec = {
                    "_id": ex.get("_id", str(i)),
                    "dataset": ds_name,
                    "prediction": pred,
                    "answers": ex["answers"],
                    "input": ex["input"],
                    "length": ex.get("length", None),
                    "qwen_prompt_tokens": int(encoded["input_ids"].shape[1]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Saved: {out_path}")