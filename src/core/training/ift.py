from concurrent.futures import ThreadPoolExecutor
from peft import PeftModel
from src.core.settings import settings
from src.core.training.common import load_training_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth.chat_templates import get_chat_template

import asyncio

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    base_model_id = load_training_config("MODEL")["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype = load_training_config("MODEL")["dtype"],
        device_map = "auto",
    )
    model = PeftModel.from_pretrained(model, settings.MODEL_PATH)

    # If merge_and_unload() used before saving:
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = settings.MODEL_PATH,
    #     max_seq_length = load_training_config("MODEL")["max_seq_length"],
    #     dtype = load_training_config("MODEL")["dtype"],
    #     load_in_4bit = load_training_config("MODEL")["load_in_4bit"]
    # )

    return model, tokenizer

async def load_model_and_tokenizer_async():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        model, tokenizer = await loop.run_in_executor(executor, load_model_and_tokenizer)
    return model, tokenizer

def chunk_tokens_ift(batch, tokenizer, max_tokens_sequence = load_training_config("MODEL")["max_seq_length"]):
    all_chunks = {"input_ids": [], "attention_mask": []}
    for text in batch["set"]:
        input_ids = tokenizer.apply_chat_template(
            text,
            tokenize = True,
            add_generation_prompt = False
        )
        if len(input_ids)> max_tokens_sequence:
            continue

        all_chunks['input_ids'].append(input_ids)
        all_chunks['attention_mask'].append([1]*len(input_ids))

    return all_chunks


# instruct_dataset = dataset.remove_columns(['key', 'text', 'summary', 'no_tokens'])


# tokenized_instruct_dataset_train = tokenize_dataset(instruct_dataset)