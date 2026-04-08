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
import random

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
def get_process_entity_dataset_resources():
    categories = list(load_cuad_resource("Entities").values())
    non_bool_categories = ["Document Name", "Parties", "Agreement Date", "Expiration Date", "Effective Date", "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law", "Warranty Duration"]
    bool_categories = [cat for cat in categories if cat not in non_bool_categories]
    categories_processed = {cat.replace(" ", ""): cat for cat in categories}
    return categories_processed, non_bool_categories, bool_categories

def process_entity_dataset(entity_dataset: Dataset):
    categories_processed, non_bool_categories, bool_categories = get_process_entity_dataset_resources()
    erroneous_entities = {}
    entity_dataset_list = []
    for i in entity_dataset:
        try:
            sample_entities = json.loads(i["entities"])["Entities"]
            sample_entity_list = []

            for entity_item in sample_entities:
                if entity_item["Entity"] not in list(categories_processed.values()):
                    entity_item["Entity"] = categories_processed[process.extractOne(entity_item["Entity"].replace(" ", ""), list(categories_processed.keys()), scorer = fuzz.token_sort_ratio)[0]]
                
                if entity_item["Entity"] in non_bool_categories:
                    pass
                elif entity_item["Entity"] in bool_categories:
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

def multipart_qa_resources():
    _, non_bool_categories, bool_categories = get_process_entity_dataset_resources()
    ques_samples = load_cuad_resource(resource = "SampleQuestions")
    return ques_samples, bool_categories, non_bool_categories

def prepare_multipart_qa(sample, ques_samples, bool_cat, non_bool_cat):
    sample["qa_pairs"] = {"pos": [], "neg": []}
    for entity in sample["entities"]:
        if entity["Entity"] in bool_cat:
            if entity["Answer"].lower().strip() in ["yes"]:
                ques = random.choice(ques_samples[entity["Entity"]])
                ans = f"{entity['Answer']}\nReference: {entity['Span']}"
                sample["qa_pairs"]["pos"].append({"entity": entity["Entity"], "question": ques, "answer": ans})
            else:
                if len(sample["qa_pairs"]["neg"]) <= 10:
                    ques = random.choice(ques_samples[entity["Entity"]])
                    sample["qa_pairs"]["neg"].append({"entity": entity["Entity"], "question": ques, "answer": "N/A"})
        elif entity["Entity"] in non_bool_cat:
            if entity["Answer"].lower().strip() not in ["n/a", "no"]:
                ques = random.choice(ques_samples[entity["Entity"]])
                ans = f"{entity['Answer']}\nReference: {entity['Span']}"
                sample["qa_pairs"]["pos"].append({"entity": entity["Entity"], "question": ques, "answer": ans})
            else:
                if len(sample["qa_pairs"]["neg"]) <= 10:
                    ques = random.choice(ques_samples[entity["Entity"]])
                    sample["qa_pairs"]["neg"].append({"entity": entity["Entity"], "question": ques, "answer": "N/A"})
    return sample


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
        for text, qa_pairs in zip(batch["text"], batch["qa_pairs"]):
            messages = [
                {"role": "system", "content": _get_prompt("clause_detection_system_prompt")["prompt"]},
                {"role": "user", "content": text},
                {"role": "assistant", "content": "How can I assist you with this contract?"}
            ]

            for qa_pair in qa_pairs["pos"]:
                messages.append({"role": "user", "content": qa_pair["question"]})
                messages.append({"role": "assistant", "content": qa_pair["answer"]})
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize = True,
                    add_generation_prompt = False
                )
                if len(input_ids) > max_tokens_sequence:
                    break

            for qa_pair in qa_pairs["neg"]:
                if len(input_ids) > max_tokens_sequence:
                    break
                messages.append({"role": "user", "content": qa_pair["question"]})
                messages.append({"role": "assistant", "content": qa_pair["answer"]})
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize = True,
                    add_generation_prompt = False
                )
            
            all_chunks['set'].append(messages)
            all_chunks['input_ids'].append(input_ids)
            all_chunks['attention_mask'].append([1]*len(input_ids))
    else:
        raise ValueError("Invalid task type. Must be 'summary' or 'qa'.")
    return all_chunks