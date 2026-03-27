from concurrent.futures import ThreadPoolExecutor
from src.core.settings import settings
from src.core.training.common import load_training_config
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

import asyncio

def get_model_and_tokenizer():
    model_config = load_training_config("Model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = MODEL_CONFIGURATIONS["model_name"],
        # max_seq_length = MAX_SEQ_LEN_CPT,
        # dtype = torch.float16,
        # load_in_4bit = MODEL_CONFIGURATIONS["load_in_4bit"],
        **model_config
    )
    peft_config = load_training_config("Peft")
    model = FastLanguageModel.get_peft_model(
        model,
        # r = MODEL_CONFIGURATIONS["lora_rank"],
        # target_modules = MODEL_CONFIGURATIONS["target_modules"],
        # lora_alpha = MODEL_CONFIGURATIONS["lora_alpha"],
        # lora_dropout = MODEL_CONFIGURATIONS["lora_dropout"],
        # bias = MODEL_CONFIGURATIONS["bias"],
        # use_gradient_checkpointing = MODEL_CONFIGURATIONS["use_gradient_checkpointing"],
        # random_state = 3407,
        # use_rslora = MODEL_CONFIGURATIONS["use_rslora"],
        # loftq_config = MODEL_CONFIGURATIONS["loftq_config"]
        **peft_config
    )

    return model, tokenizer

async def get_model_and_tokenizer_async():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        model, tokenizer = await loop.run_in_executor(executor, get_model_and_tokenizer)
    return model, tokenizer

def chunk_tokens_cpt(batch, tokenizer, overlap = 128):
    all_chunks = {"input_ids": [], "attention_mask": []}
    max_tokens_sequence = load_training_config("TrainingArgsCPT")["max_seq_length"]
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