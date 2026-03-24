from .common import create_and_submit_batch_job, download_shuffled_samples, _add_key_column, poll_and_store_batch_results, get_text_splitter, _get_gemini_client, _get_prompt, calculate_token_count
from datasets import Dataset, load_from_disk
from pathlib import Path
from src.core.settings import settings

import asyncio
import json

def split_training_samples(batch):
    all_texts = []
    all_keys = []
    all_llama_tokens = []
    text_splitter = get_text_splitter()

    for i in range(len(batch["text"])):
        sample_text = batch["text"][i]
        sample_key = batch["key"][i]
        
        split_docs = text_splitter.create_documents(texts = [sample_text], metadatas = [{"key": sample_key}])
        
        for doc in split_docs:
            if doc.metadata["llama_tokens"] >= 1500:
                all_texts.append(doc.page_content)
                all_keys.append(doc.metadata["key"])
                all_llama_tokens.append(doc.metadata["llama_tokens"])
                
    return {
        "text": all_texts,
        "key": all_keys,
        "text_llama_tokens": all_llama_tokens,
    }

def download_contracts_dataset(seed, split = True):
    dataset = download_shuffled_samples(
        dataset_name = "joelniklaus/Multi_Legal_Pile", 
        config = "en_contracts", 
        n_samples = settings.INS_FT_MLP_CONTRACT_SAMPLE_SIZE, 
        general = False,
        seed = seed
    )
    dataset = _add_key_column(dataset)
    print("Saving contracts dataset to %s", settings.INS_FT_DATA_DIR)
    if split:
        dataset = dataset.map(
            split_training_samples,
            batched=True,
            remove_columns=dataset.column_names
        )
    dataset.save_to_disk(settings.INS_FT_DATA_DIR)
    return dataset

def load_contracts_dataset(limit = None):
    if not settings.INS_FT_DATA_DIR.exists():
        raise FileNotFoundError("Keyed contracts dataset not found. Run download_contracts_dataset() first.")
    try:
        dataset = load_from_disk(settings.INS_FT_DATA_DIR)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return dataset
    except Exception as e:
        raise RuntimeError(f"Error loading contracts dataset: {e}")


async def parse_results(
        job_name,
        text_column_name: str,
        text_column_token_count_name: str,
        batch_results_path: Path,
        key_column = "key",
        dataset_limit = None
    ):

    lines, unable_to_parse = await poll_and_store_batch_results(job_name)
    contracts = load_contracts_dataset(limit = dataset_limit)
    # key_to_text = {entry[key_column]: entry["text"] for entry in contracts}
    records = {}
    errorneous_keys = {}
    for line in lines:
        key = line["key"]
        parts = line.get("response", {}).get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0]
        if parts == {}:
            errorneous_keys[key] = f"ErrorInResponse"
        text = parts.get("text", "N/A")
        token_count = calculate_token_count(text, "llama") if text != "N/A" else 0
        records[key] = {
            text_column_name: text,
            text_column_token_count_name: token_count
        }
        try:
            json.loads(text)
        except Exception as e:
            errorneous_keys[key] = f"ResponseJSONDecodeError: {e}"
            continue

    def add_fields(batch):
        response_texts = []
        response_texts_tokens = []
        
        for k in batch["key"]:
            entry = records.get(k)
            if entry:
                response_texts.append(entry[text_column_name])
                response_texts_tokens.append(entry[text_column_token_count_name])
            else:
                response_texts.append("N/A")
                response_texts_tokens.append(0)
        
        return {
            text_column_name: response_texts,
            text_column_token_count_name: response_texts_tokens,
        }

    results_dataset = contracts.map(add_fields, batched=True)

    # results_dataset = Dataset.from_list(records)
    batch_results_path.mkdir(parents=True, exist_ok=True)
    print("Saving batch results to %s", batch_results_path)
    await asyncio.to_thread(results_dataset.save_to_disk, batch_results_path)
    return results_dataset, errorneous_keys

################################################################ Summarization ################################################################

def get_summarization_resources():
    prompt_template = _get_prompt("contract_summarization")["prompt"]
    client = _get_gemini_client()

    SUMMARIZATION_RESPONSE_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "summary": {
                "type": "STRING",
                "description": "A clear, structured summary of the contract."
            }
        },
        "required": ["summary"]
    }

    return prompt_template, client, SUMMARIZATION_RESPONSE_SCHEMA

async def submit_summarization_batch_job(dataset: Dataset, key_column=None, display_name="MLP_Contract_Summarization_Batch", submit = True):
    prompt_template, client, schema = get_summarization_resources()
    entries = []
    key_column = "key" if not key_column else key_column
    for contract in dataset:
        entries.append({
            "key": contract[key_column],
            "method": "generateContent",
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": prompt_template.replace("{{CONTRACT_TEXT}}", contract["text"])
                    }]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    "thinkingConfig": {"thinkingLevel": "MINIMAL"}
                }
            }
        })

    job = await create_and_submit_batch_job(
        client = client,
        entries = entries,
        save_batch_path = settings.INS_FT_MLP_DATA_DIR_SUMM / settings.INS_FT_MLP_BATCH_INPUT,
        batch_file_name = str(settings.INS_FT_MLP_BATCH_INPUT),
        batch_job_name = display_name,
        submit = submit
    )
    return job

################################################################ Multi-part QnA ################################################################

def get_qna_resources():
    prompt_template = _get_prompt("entity_extraction")["prompt"]
    client = _get_gemini_client()
    
    QNA_RESPONSE_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "Entities": {
                "type": "ARRAY",
                "description": "A list of entities to check in the contract, with their extracted answers and text spans.",
                "minItems": 41,
                "maxItems": 41,
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "Entity": {
                            "type": "STRING",
                            "description": "The name of the entity being checked."
                        },
                        "Answer": {
                            "type": "STRING",
                            "description": "The extracted answer or value for the entity in the specified format."
                        },
                        "Span": {
                            "type": "STRING",
                            "description": "The exact text span from the contract where the entity is found/extracted."
                        }
                    },
                    "required": ["Entity", "Answer", "Span"]
                },

            }
        },
        "required": ["Entities"]
    }
    return prompt_template, client, QNA_RESPONSE_SCHEMA

async def submit_qna_batch_job(dataset: Dataset, key_column=None, display_name="MLP_Contract_QnA_Batch", submit = True):
    prompt_template, client, schema = get_qna_resources()
    entries = []
    if not key_column:
        key_column = "key" if "key" in dataset.column_names else "section_key"

    with open(f"training_data\instruction_ft_data\cuad\ques_samples.json", "r", encoding="utf-8") as f:
        ques_samples = json.load(f)
    with open(f"training_data\instruction_ft_data\multi_legal_pile\category-description.json", "r", encoding="utf-8") as f:
        category_description = json.load(f)

    categories = list(ques_samples.keys())
    ####
    # non_bool_cat = ["Document Name", "Parties", "Agreement Date", "Expiration Date", "Effective Date", "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law"]
    ####
    categories_str = ",\n".join([str(_) for _ in category_description if _["Entity"] in categories])

    for contract in dataset:
        entries.append({
            "key": contract[key_column],
            "method": "generateContent",
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": prompt_template.replace("{{ENTITIES_TO_CHECK}}", categories_str).replace("{{CONTRACT_TEXT}}", contract["text"])
                    }]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    # "thinkingConfig": {"thinkingLevel": "MINIMAL"}
                }
            }
        })

    job = await create_and_submit_batch_job(
        client = client,
        entries = entries,
        save_batch_path = settings.INS_FT_MLP_DATA_DIR_QNA / settings.INS_FT_MLP_BATCH_INPUT,
        batch_file_name = str(settings.INS_FT_MLP_BATCH_INPUT),
        batch_job_name = display_name,
        submit = submit
    )
    
    return job

def _extract_entities(line: dict):
    response = line.get("response") or {}
    usage = response.get("usageMetadata") or {}
    candidates = response.get("candidates") or []
    parts = (
        (candidates[0].get("content") or {}).get("parts") if candidates else []
    )
    text = (parts[0].get("text") if parts else "") or ""
    if not text:
        return ""
    try:
        payload = json.loads(text)
        entities = payload.get("Entities", "")
    except json.JSONDecodeError:
        entities = text
    return entities

async def poll_and_download_qna_results(job_name, key_column = "key", dataset_limit = None):
    lines, unable_to_parse = await poll_and_store_batch_results(job_name)
    contracts = load_contracts_dataset(limit = dataset_limit)
    key_to_text = {entry[key_column]: entry["text"] for entry in contracts}
    records = []
    for line in lines:
        key = line.get("key")
        entities = _extract_entities(line)
        records.append({
            "key": key,
            "text": key_to_text.get(key, ""),
            "entities": entities,
            "entities_llama_tokens": calculate_token_count(str(entities))
        })

    results_dataset = Dataset.from_list(records)
    batch_results_path = settings.INS_FT_MLP_DATA_DIR_QNA / settings.INS_FT_MLP_BATCH_RESULTS
    batch_results_path.mkdir(parents=True, exist_ok=True)
    print("Saving batch results to %s", batch_results_path)
    await asyncio.to_thread(results_dataset.save_to_disk, batch_results_path)
    return results_dataset