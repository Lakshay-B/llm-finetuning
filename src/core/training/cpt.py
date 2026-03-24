from concurrent.futures import ThreadPoolExecutor
from src.core.settings import settings
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

import asyncio
import torch

MAX_SEQ_LEN_CPT = settings.MAX_SEQ_LEN_CPT

MODEL_CONFIGURATIONS = {
   "model_name": "unsloth/Llama-3.2-3B",
   "load_in_4bit": True,
   "lora_rank": 128,
   "target_modules": ["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 
                      "embed_tokens", "lm_head"],
    "lora_alpha" : 32,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": True,
    "use_rslora": True,
    "loftq_config": None
}

def get_model_and_tokenizer():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_CONFIGURATIONS["model_name"],
        max_seq_length = MAX_SEQ_LEN_CPT,
        dtype = torch.float16,
        load_in_4bit = MODEL_CONFIGURATIONS["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = MODEL_CONFIGURATIONS["lora_rank"],
        target_modules = MODEL_CONFIGURATIONS["target_modules"],
        lora_alpha = MODEL_CONFIGURATIONS["lora_alpha"],
        lora_dropout = MODEL_CONFIGURATIONS["lora_dropout"],
        bias = MODEL_CONFIGURATIONS["bias"],
        use_gradient_checkpointing = MODEL_CONFIGURATIONS["use_gradient_checkpointing"],
        random_state = 3407,
        use_rslora = MODEL_CONFIGURATIONS["use_rslora"],
        loftq_config = MODEL_CONFIGURATIONS["loftq_config"]
    )

    return model, tokenizer

async def get_model_and_tokenizer_async():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        model, tokenizer = await loop.run_in_executor(executor, get_model_and_tokenizer)
    return model, tokenizer

def chunk_tokens(batch, tokenizer, max_tokens_sequence = MAX_SEQ_LEN_CPT, overlap = 128):
    all_chunks = {"input_ids": [], "attention_mask": []}
    stride = max_tokens_sequence-overlap
    for text in batch["text"]:
        input_ids = tokenizer(
            text,
            add_special_tokens=False
        )['input_ids']

        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i:i+max_tokens_sequence]

            if len(chunk) < 128:  # drop tiny fragments
                continue

            all_chunks['input_ids'].append(chunk)
            all_chunks['attention_mask'].append([1]*len(chunk))

    return all_chunks

def tokenize_dataset(ds, tokenizer):
    return ds.map(
        chunk_tokens,
        batched=True,
        batch_size=1,
        remove_columns=ds.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_tokens_sequence": MAX_SEQ_LEN_CPT,
        },
        desc="Chunking dataset",
    )