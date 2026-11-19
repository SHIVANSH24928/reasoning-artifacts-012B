"""
Model Inference Pipeline
========================
Paper Section 3.3: "Compute and Execution Environment"
Paper Section 4: "Finding 1: Implementation Hygiene in Batched Decoder Inference"

Configured with:
- Greedy decoding (temperature=0.0, do_sample=False)
- max_new_tokens=1024
- 4-bit quantization with FP4 compute (load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
- Explicit left-padding for causal decoder-only generation
"""

import torch
from typing import List, Optional
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    pass  # Allow importing for offline analysis without GPU


def load_model_and_tokenizer(model_name: str, padding_side: str = "left", device_map: str = "auto"):
    """
    Loads open-weight model with 4-bit quantization and specified padding side.
    
    Args:
        model_name: HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')
        padding_side: 'left' (recommended for decoder inference) or 'right' (flawed default)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="fp4"  # Default in bitsandbytes
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16
    )
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generates deterministic response using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only the generated completion tokens
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
