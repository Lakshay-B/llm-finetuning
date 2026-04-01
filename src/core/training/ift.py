from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset
from functools import lru_cache
from peft import PeftModel
from rapidfuzz import process, fuzz
from src.core.settings import settings
from src.core.training.common import load_training_config
from src.get_training_data.common import _get_prompt
from src.get_training_data.cuad_data import load_cuad_resource
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from unsloth.chat_templates import get_chat_template

import asyncio
import json

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

def process_summary_dataset(sample):
    try:
        sample["summary"] = json.loads(sample["summary"]).get("summary")  # In case the summary is wrapped in a dict with a "summary" key
        return sample
    except json.JSONDecodeError:
        return sample 

@lru_cache()
def process_entity_dataset_resources():
    categories = list(load_cuad_resource("Entities").values())
    non_bool_category = ["Document Name", "Parties", "Agreement Date", "Expiration Date", "Effective Date", "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law", "Warranty Duration"]
    bool_category = [cat for cat in categories if cat not in non_bool_category]
    categories_processed = {cat.replace(" ", ""): cat for cat in categories}
    return categories_processed, non_bool_category, bool_category

def process_entity_dataset(entity_dataset: Dataset):
    categories_processed, non_bool_category, bool_category = process_entity_dataset_resources()
    erroneous_entities = {}
    entity_dataset_list = []
    for i in entity_dataset:
        try:
            sample_entities = json.loads(i["entities"])["Entities"]
            sample_entity_list = []

            for entity_item in sample_entities:
                if entity_item["Entity"] not in list(categories_processed.values()):
                    entity_item["Entity"] = categories_processed[process.extractOne(entity_item["Entity"].replace(" ", ""), list(categories_processed.keys()), scorer = fuzz.token_sort_ratio)[0]]
                
                if entity_item["Entity"] in non_bool_category:
                    pass
                elif entity_item["Entity"] in bool_category:
                    if entity_item["Answer"].lower().strip() in ["n/a", "n o", "n / a"]:
                        entity_item["Answer"] = "No"
                    elif entity_item["Answer"].lower().strip() == "y e s":
                        entity_item["Answer"] = "Yes"
                    else:
                        pass

                    if entity_item["Answer"].lower().strip() not in ["yes", "no"]:
                        erroneous_entities[i["key"]] = {"error": "BoolAnswerError", "content": entity_item['Answer']}
                else:
                    erroneous_entities[i["key"]] = {"error": "CategoryError", "content": f"{i['key']}: {entity_item['Entity']}"}

                sample_entity_list.append(entity_item)
            
            entity_dataset_list.append({
                "key": i["key"],
                "text": i["text"],
                "entities": sample_entity_list,
                "text_llama_tokens": i["text_llama_tokens"],
                "entities_llama_tokens": i["entities_llama_tokens"]
            })

        except json.JSONDecodeError:
            erroneous_entities[i["key"]] = {"error": "EntitiesJSONDecodeError", "content": i["entities"]}

    return Dataset.from_list(entity_dataset_list), erroneous_entities


def chunk_tokens_ift(batch, tokenizer, task: Literal["summary", "qa"]):
    all_chunks = {"set": [], "input_ids": [], "attention_mask": []}
    max_tokens_sequence = load_training_config("MODEL")["max_seq_length"]
    if task == "summary":
        for text, summary in zip(batch["text"], batch["summary"]):
            messages = [
                {"role": "system", "content": _get_prompt("summary_system_prompt")["prompt"]},
                {"role": "user", "content": text},
                {"role": "assistant", "content": summary}
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = False
            )
            if len(input_ids)> max_tokens_sequence:
                continue
            
            all_chunks['set'].append(messages)
            all_chunks['input_ids'].append(input_ids)
            all_chunks['attention_mask'].append([1]*len(input_ids))
            
    elif task == "qa":
        for text, entities in zip(batch["text"], batch["entities"]):
            messages = [
                {"role": "system", "content": _get_prompt("clause_detection_system_prompt")["prompt"]},
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps({"Entities": entities})}   ### edit here
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = False
            )
            if len(input_ids)> max_tokens_sequence:
                continue
            
            all_chunks['set'].append(messages)
            all_chunks['input_ids'].append(input_ids)
            all_chunks['attention_mask'].append([1]*len(input_ids))
    else:
        raise ValueError("Invalid task type. Must be 'summary' or 'qa'.")
    return all_chunks