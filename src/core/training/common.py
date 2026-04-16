"""
common.py
---------
Shared utilities for loading training configuration and tokenising datasets.
"""

from typing import Literal

import os
import torch
import yaml

dtype_mapping = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32
}

def load_training_config(
    name: Literal["Model", "Peft", "TrainingArgsCPT", "TrainerIFT", "TrainingArgsIFT"]
) -> dict | None:
    """Load and return a named section of the training configuration YAML.

    Args:
        name: The config section to retrieve. Must be one of: ``"Model"``, ``"Peft"``, ``"TrainingArgsCPT"``, ``"TrainerIFT"``, ``"TrainingArgsIFT"``.

    Returns:
        The requested config dict, or ``None`` if the runtime environment is
        not recognised.
    """
    with open(
        os.path.join("src", "core", "training", "training_config.yaml"),
        encoding="utf-8",
    ) as f:
        training_config = yaml.safe_load(f)

    if os.getenv("RUNTIME_ENV") == "test":
        training_config = training_config["TEST"]
    elif os.getenv("RUNTIME_ENV") == "gpu":
        training_config = training_config["GPU"]
    else:
        return None

    training_config["MODEL_CONFIG"]["dtype"] = dtype_mapping[
        training_config["MODEL_CONFIG"]["dtype"]
    ]

    if name == "Model":
        config = training_config["MODEL_CONFIG"]
    elif name == "Peft":
        config = training_config["PEFT_CONFIG"]
    elif name == "TrainingArgsCPT":
        config = training_config["CPT_TRAINING_ARGS"]
    elif name == "TrainerIFT":
        config = training_config["IFT_TRAINER_ARGS"]
    elif name == "TrainingArgsIFT":
        config = training_config["IFT_TRAINING_ARGS"]
    else:
        raise ValueError(f"Unknown training config section: '{name}'")

    return config

def tokenize_dataset(ds, tokenizer, chunking_function: callable):
    """Apply a chunking function to an entire dataset using batched mapping.

    Args:
        ds:                The ``Dataset`` to tokenise.
        tokenizer:         The tokeniser instance forwarded to
                           ``chunking_function`` via ``fn_kwargs``.
        chunking_function: A callable with signature
                           ``(batch: dict, tokenizer) -> dict`` that splits
                           and tokenises a batch of texts.

    Returns:
        A new ``Dataset`` containing only the columns produced by
        ``chunking_function`` (typically ``"input_ids"`` and
        ``"attention_mask"``).
    """
    return ds.map(
        chunking_function,
        batched=True,
        batch_size=1,
        remove_columns=ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Chunking dataset",
    )