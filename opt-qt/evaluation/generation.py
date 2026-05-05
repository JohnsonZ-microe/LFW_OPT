"""
Text Generation Evaluation for OPT Models.

This module provides text generation functionality for evaluating
quantized models qualitatively.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import GenerationConfig


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    device: str = "cuda"
) -> List[str]:
    """
    Generate text from a prompt.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate
        device: Device to run generation on
        
    Returns:
        List of generated text strings
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
    
    return generated_texts


def compare_generation(
    original_model,
    quantized_model,
    tokenizer,
    prompts: List[str],
    max_length: int = 100,
    temperature: float = 1.0,
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Compare text generation between original and quantized models.
    
    Args:
        original_model: Original (non-quantized) model
        quantized_model: Quantized model
        tokenizer: Tokenizer for the models
        prompts: List of input prompts
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        device: Device to run generation on
        
    Returns:
        List of dictionaries with comparison results
    """
    results = []
    
    print("="*80)
    print("Comparing Text Generation: Original vs Quantized")
    print("="*80)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt}")
        print("-"*80)
        
        # Generate with original model
        print("Generating with original model...")
        original_texts = generate_text(
            original_model,
            tokenizer,
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=False,  # Use greedy for comparison
            device=device
        )
        
        # Generate with quantized model
        print("Generating with quantized model...")
        quantized_texts = generate_text(
            quantized_model,
            tokenizer,
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=False,  # Use greedy for comparison
            device=device
        )
        
        # Print results
        print("\nOriginal Model Output:")
        print(original_texts[0])
        print("\nQuantized Model Output:")
        print(quantized_texts[0])
        print("-"*80)
        
        results.append({
            "prompt": prompt,
            "original_output": original_texts[0],
            "quantized_output": quantized_texts[0]
        })
    
    print("\n" + "="*80 + "\n")
    
    return results


class GenerationEvaluator:
    """
    Text generation evaluator class.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Initialize generation evaluator.
        
        Args:
            model: Model to use for generation
            tokenizer: Tokenizer for the model
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text strings
        """
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_length,
            temperature,
            top_k,
            top_p,
            do_sample,
            num_return_sequences,
            self.device
        )
    
    def interactive_generation(self):
        """
        Interactive text generation loop.
        """
        print("="*80)
        print("Interactive Text Generation")
        print("="*80)
        print("Enter prompts to generate text. Type 'quit' to exit.")
        print("-"*80)
        
        while True:
            prompt = input("\nPrompt: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive generation.")
                break
            
            if not prompt.strip():
                continue
            
            print("\nGenerating...")
            outputs = self.generate(prompt, do_sample=True, num_return_sequences=1)
            
            print("\nGenerated Text:")
            print("-"*80)
            print(outputs[0])
            print("-"*80)

