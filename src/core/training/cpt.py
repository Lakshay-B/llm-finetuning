"""
cpt.py
------
Continued pre-training (CPT) utilities: model/tokeniser loading and
sliding-window tokenisation for long-document training.
"""

from concurrent.futures import ThreadPoolExecutor
from src.core.training.common import load_training_config
from unsloth import FastLanguageModel

import asyncio

def get_model_and_tokenizer():
    """Load a Unsloth ``FastLanguageModel`` base model and attach a LoRA adapter.

    Returns:
        A ``(model, tokenizer)`` tuple with the LoRA adapter applied and ready for continued pre-training.
    """
    model_config = load_training_config("Model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        **model_config
    )
    peft_config = load_training_config("Peft")
    model = FastLanguageModel.get_peft_model(
        model,
        **peft_config
    )

    return model, tokenizer

async def get_model_and_tokenizer_async():
    """Async wrapper around ``get_model_and_tokenizer``.

    Returns:
        A ``(model, tokenizer)`` tuple, as returned by
        ``get_model_and_tokenizer``.
    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        model, tokenizer = await loop.run_in_executor(executor, get_model_and_tokenizer)
    return model, tokenizer

def chunk_tokens_cpt(batch: dict, tokenizer, overlap: int = 128) -> dict:
    """Tokenise a batch of texts using a sliding window and return fixed-length
    token chunks suitable for causal language model training.

    Args:
        batch:    A batched dict from ``Dataset.map`` containing a ``"text"`` list of raw contract strings.
        tokenizer: The model tokeniser used to convert text to token IDs. Special tokens are not added (``add_special_tokens=False``).
        overlap:  Number of tokens shared between consecutive chunks (default 128). Also used as the minimum chunk length threshold — chunks shorter than this are dropped.

    Returns:
        A dict with two equal-length lists:
        - ``"input_ids"``:      List of token-ID lists, each at most
                               ``max_seq_length`` tokens long.
        - ``"attention_mask"``: All-ones mask matching each ``input_ids`` entry.
    """
    all_chunks: dict = {"input_ids": [], "attention_mask": []}
    max_tokens_sequence = load_training_config("TrainingArgsCPT")["max_seq_length"]
    stride = max_tokens_sequence - overlap

    for text in batch["text"]:
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i: i + max_tokens_sequence]
            if len(chunk) < overlap:
                continue

            all_chunks["input_ids"].append(chunk)
            all_chunks["attention_mask"].append([1] * len(chunk))

    return all_chunks