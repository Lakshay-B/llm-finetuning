from typing import Literal

import os
import torch
import yaml

dtype_mapping = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32
}

def load_training_config(name: Literal["Model", "Peft", "TrainingArgsCPT", "TrainerIFT", "TrainingArgsIFT"]):
    with open(os.path.join("src", "core", "training", "training_config.yaml")) as f:
        training_config = yaml.safe_load(f)
    if os.getenv('RUNTIME_ENV') == 'test':
        training_config = training_config['TEST']
    elif os.getenv('RUNTIME_ENV') == 'gpu':
        training_config = training_config['GPU']
    else:
        return None
    
    training_config["MODEL_CONFIG"]["dtype"] = dtype_mapping[training_config["MODEL_CONFIG"]["dtype"]]
    if name == "Model":
        CONFIGURATIONS = training_config["MODEL_CONFIG"]
    if name == "Peft":
        CONFIGURATIONS = training_config["PEFT_CONFIG"]
    if name == "TrainingArgsCPT":
        CONFIGURATIONS = training_config["CPT_TRAINING_ARGS"]
    if name == "TrainerIFT":
        CONFIGURATIONS = training_config["IFT_TRAINER_ARGS"]
    if name == "TrainingArgsIFT":
        CONFIGURATIONS = training_config["IFT_TRAINING_ARGS"]
    return CONFIGURATIONS

def tokenize_dataset(ds, tokenizer, chunking_function: callable):
    return ds.map(
        chunking_function,
        batched=True,
        batch_size=1,
        remove_columns=ds.column_names,
        fn_kwargs={
            "tokenizer": tokenizer
        },
        desc="Chunking dataset",
    )