"""
Perplexity Evaluation for OPT Models.

This module provides perplexity evaluation functionality based on the
AWQ evaluation method.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any
import tqdm


def _load_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    streaming: bool = False,
):
    if dataset_config is None:
        return load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
        )
    else:
        return load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )


def _collect_tokens_from_iterable(
    dataset,
    tokenizer,
    text_column: str = "text",
    max_eval_tokens: int = 65536,
    min_doc_tokens: int = 128,
    device: str = "cuda",
):
    token_ids = []
    eos_id = tokenizer.eos_token_id

    for ex in dataset:
        if text_column not in ex:
            raise KeyError(
                f"text_column='{text_column}' not found. "
                f"Available columns: {list(ex.keys())}"
            )

        text = ex[text_column]
        if text is None:
            continue

        text = str(text).strip()
        if not text:
            continue

        ids = tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

        if len(ids) < min_doc_tokens:
            continue

        token_ids.extend(ids)

        if eos_id is not None:
            token_ids.append(eos_id)

        if len(token_ids) >= max_eval_tokens:
            token_ids = token_ids[:max_eval_tokens]
            break

    if len(token_ids) == 0:
        raise RuntimeError("No tokens collected from dataset.")

    return torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-2-raw-v1",
    split: str = "test",
    seq_length: int = 2048,
    device: str = "cuda",
    stride: Optional[int] = None,
    text_column: str = "text",
    streaming: bool = False,
    max_eval_tokens: Optional[int] = None,
    min_doc_tokens: int = 128,
) -> float:
    """
    Evaluate perplexity on a dataset.

    Supports both normal HF datasets and streaming datasets such as FineWeb.
    """

    print(
        f"Evaluating perplexity on {dataset_name} "
        f"({dataset_config}, {split}), streaming={streaming}, "
        f"text_column={text_column}"
    )

    dataset = _load_hf_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        streaming=streaming,
    )

    if streaming:
        if max_eval_tokens is None:
            max_eval_tokens = 65536

        testenc = _collect_tokens_from_iterable(
            dataset=dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            max_eval_tokens=max_eval_tokens,
            min_doc_tokens=min_doc_tokens,
            device=device,
        )
    else:
        texts = []
        for ex in dataset:
            if text_column not in ex:
                raise KeyError(
                    f"text_column='{text_column}' not found. "
                    f"Available columns: {list(ex.keys())}"
                )

            text = ex[text_column]
            if text is None:
                continue

            text = str(text).strip()
            if text:
                texts.append(text)

        if len(texts) == 0:
            raise RuntimeError("No valid text found in evaluation dataset.")

        testenc = tokenizer(
            "\n\n".join(texts),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)

        if max_eval_tokens is not None:
            testenc = testenc[:, :max_eval_tokens]

    model.seqlen = seq_length

    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise RuntimeError(
            f"Not enough tokens for evaluation: "
            f"num_tokens={testenc.numel()}, seq_length={model.seqlen}"
        )

    testenc = testenc[:, : nsamples * model.seqlen]

    model = model.eval()
    nlls = []
    total_tokens = 0

    print(
        f"Collected tokens={testenc.numel()}, "
        f"seq_length={model.seqlen}, nsamples={nsamples}"
    )

    loss_fct = nn.CrossEntropyLoss()

    for i in tqdm.tqdm(range(nsamples), desc="Evaluating perplexity"):
        batch = testenc[
            :,
            i * model.seqlen : (i + 1) * model.seqlen,
        ].to(device)

        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        ntokens = shift_labels.numel()
        neg_log_likelihood = loss.float() * ntokens

        nlls.append(neg_log_likelihood)
        total_tokens += ntokens

        if i == 0 or torch.isnan(neg_log_likelihood) or torch.isinf(neg_log_likelihood):
            print(
                f"  Sample {i}: loss={loss.item():.4f}, "
                f"nll={neg_log_likelihood.item():.4f}"
            )

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)

    if torch.isnan(ppl) or torch.isinf(ppl):
        print(f"  WARNING: PPL is {ppl.item()}")
        print(f"  Total NLL: {torch.stack(nlls).sum().item()}")
        print(f"  Total tokens: {total_tokens}")

    print(f"Perplexity: {ppl.item():.4f}")

    return ppl.item()
    
def evaluate_perplexity_sliding_window(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    seq_length: int = 2048,
    stride: int = 512,
    device: str = "cuda"
) -> float:
    """
    Evaluate perplexity using sliding window approach.
    
    This provides more accurate perplexity estimation by using overlapping windows.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        split: Dataset split
        seq_length: Sequence length for evaluation
        stride: Stride for sliding window
        device: Device to run evaluation on
        
    Returns:
        Perplexity value
    """
    print(f"Evaluating perplexity with sliding window (stride={stride})...")
    
    # Load dataset
    testenc = load_dataset(dataset_name, dataset_config, split=split)
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    testenc = testenc.input_ids.to(device)
    
    # Set model to evaluation mode
    model = model.eval()
    
    # Calculate perplexity with sliding window
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm.tqdm(
        range(0, testenc.size(1), stride),
        desc="Evaluating with sliding window"
    ):
        end_loc = min(begin_loc + seq_length, testenc.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_ids = testenc[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == testenc.size(1):
            break
    
    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    print(f"Perplexity (sliding window): {ppl.item():.4f}")
    
    return ppl.item()


def compare_perplexity(
    original_model: nn.Module,
    quantized_model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    seq_length: int = 2048,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compare perplexity between original and quantized models.
    
    Args:
        original_model: Original (non-quantized) model
        quantized_model: Quantized model
        tokenizer: Tokenizer for the models
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        split: Dataset split
        seq_length: Sequence length for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with perplexity values and degradation
    """
    print("="*80)
    print("Comparing Perplexity: Original vs Quantized")
    print("="*80)
    
    # Evaluate original model
    print("\n[1/2] Evaluating original model...")
    original_ppl = evaluate_perplexity(
        original_model,
        tokenizer,
        dataset_name,
        dataset_config,
        split,
        seq_length,
        device
    )
    
    # Evaluate quantized model
    print("\n[2/2] Evaluating quantized model...")
    quantized_ppl = evaluate_perplexity(
        quantized_model,
        tokenizer,
        dataset_name,
        dataset_config,
        split,
        seq_length,
        device
    )
    
    # Calculate degradation
    ppl_degradation = quantized_ppl - original_ppl
    ppl_degradation_percent = (ppl_degradation / original_ppl) * 100
    
    # Print results
    print("\n" + "="*80)
    print("Perplexity Comparison Results")
    print("="*80)
    print(f"Original Model PPL:    {original_ppl:.4f}")
    print(f"Quantized Model PPL:   {quantized_ppl:.4f}")
    print(f"PPL Degradation:       {ppl_degradation:.4f} ({ppl_degradation_percent:+.2f}%)")
    print("="*80 + "\n")
    
    return {
        "original_ppl": original_ppl,
        "quantized_ppl": quantized_ppl,
        "ppl_degradation": ppl_degradation,
        "ppl_degradation_percent": ppl_degradation_percent
    }


class PerplexityEvaluator:
    """
    Perplexity evaluator class for convenient evaluation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Initialize perplexity evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        dataset_name: str = "wikitext",
        dataset_config: Optional[str] = "wikitext-2-raw-v1",
        split: str = "test",
        seq_length: int = 2048,
        use_sliding_window: bool = False,
        stride: int = 512,
        text_column: str = "text",
        streaming: bool = False,
        max_eval_tokens: Optional[int] = None,
        min_doc_tokens: int = 128,
    ) -> float:
        if use_sliding_window:
            raise NotImplementedError(
                "Streaming sliding-window PPL is not implemented yet. "
                "Use use_sliding_window=False for FineWeb."
            )

        return evaluate_perplexity(
            self.model,
            self.tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            seq_length=seq_length,
            device=self.device,
            text_column=text_column,
            streaming=streaming,
            max_eval_tokens=max_eval_tokens,
            min_doc_tokens=min_doc_tokens
        )