from datasets import Dataset, load_dataset
from datetime import datetime
from google import genai
from google.genai import types
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from src.core.settings import settings
from transformers import AutoTokenizer

import asyncio
import json
import tiktoken
import time
import unicodedata
import yaml

_TOKENIZERS = {
    "gpt-5": tiktoken.get_encoding("o200k_base"),
    "llama": AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
}

with open(os.path.join("src", "prompts.yaml")) as f:
    PROMPTS = yaml.safe_load(f)

def _get_prompt(prompt):
    return PROMPTS[prompt]

def _get_gemini_client():
    return genai.Client(
        vertexai=True,
        project=settings.PROJECT_ID,
        location="global"
    )

def _load_stream(dataset_name, config, split="train"):
    return load_dataset(dataset_name, config, split=split, streaming=True)

def _format_unicode_example(sample):
    sample["text"] = unicodedata.normalize("NFKC", sample["text"].strip())
    return sample

def _add_key_column(dataset: Dataset) -> Dataset:
    return dataset.map(lambda ex, idx: {"key": f"contract_{idx}", **ex}, with_indices=True)

def download_shuffled_samples(dataset_name, config, n_samples, general=False, seed = 42):
    stream = _load_stream(dataset_name, config)
    stream = stream.shuffle(buffer_size=settings.MLP_BUFFER_SIZE, seed=seed)
    collected = []
    for ex in stream:
        if general or ex.get("jurisdiction") == "US":
            if ".jpg" in ex.get("text"):
                continue
            collected.append(_format_unicode_example(ex))
            if len(collected) >= n_samples:
                break
    dataset = Dataset.from_list(collected)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    return dataset

def calculate_token_count(text, model_name):
    enc = _TOKENIZERS.get(model_name)
    tokens = enc.encode(text)
    return len(tokens)

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    print(f"Writing batch input to {path}")
    with open(path, "w", encoding="utf-8") as output:
        for entry in entries:
            output.write(json.dumps(entry) + "\n")
    print(f"Saved batch input to {path}")

async def create_and_submit_batch_job(client, entries: list[dict], save_batch_path, batch_file_name, batch_job_name: str, submit: bool):
    save_batch_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_write_jsonl, save_batch_path, entries)

    blob_name=f"{datetime.now().strftime('%Y-%m-%d::%H:%M:%S')}/{batch_file_name}"
    
    await asyncio.to_thread(
        _upload_to_bucket,
        bucket_name=settings.INS_FT_MLP_BUCKET_NAME, 
        blob_name=blob_name,
        jsonL=True,
        jsonL_string= "\n".join(json.dumps(entry) for entry in entries)
    )

    print(f"Uploaded to bucket: {settings.INS_FT_MLP_BUCKET_NAME}/{blob_name}")
    
    if submit:
        job = client.batches.create(
            model="gemini-3-flash-preview",
            src=f"gs://{settings.INS_FT_MLP_BUCKET_NAME}/{blob_name}",
            config=types.CreateBatchJobConfig(
                display_name=batch_job_name,
                dest=f"gs://{settings.INS_FT_MLP_BUCKET_NAME}/{settings.INS_FT_MLP_BATCH_RESULTS}/"
            )
        )

        print(f"Created batch job {job.name}")
        return job
    else:
        print(f"Batch job creation skipped for: {settings.INS_FT_MLP_BUCKET_NAME}/{blob_name}")
        return None

async def poll_and_store_batch_results(job_name, poll_interval=30, timeout=None):
    client = _get_gemini_client()
    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    start = time.time()
    batch_job = client.batches.get(name = job_name)
    while batch_job.state.name not in completed_states:
        if timeout and time.time() - start > timeout:
            raise TimeoutError("Batch job polling exceeded timeout")
        await asyncio.sleep(poll_interval)
        batch_job = await asyncio.to_thread(client.batches.get, name=job_name)
        print("Batch job %s state=%s", job_name, batch_job.state.name)
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job finished with state {batch_job.state.name}")

    decoded_content = _download_from_bucket(
        bucket_name = settings.INS_FT_MLP_BUCKET_NAME,
        blob_name = rf"batch_results/prediction-model-{batch_job.create_time.strftime('%Y-%m-%dT%H:%M')}",
        partial=True,
        # destination_path = None
    )

    # file_content = await asyncio.to_thread(
    #     client.files.download,
    #     file=batch_job.dest.file_name,
    # )
    # decoded_content = file_content.decode("utf-8")
    lines = []
    unable_to_parse = []
    for line in decoded_content.splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            unable_to_parse.append(line)
    return lines, unable_to_parse


def _upload_to_bucket(bucket_name: str, blob_name: str, file_path = None, jsonL=False, jsonL_string=None):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if jsonL:
        blob.upload_from_string(jsonL_string, content_type="application/x-jsonlines")
    else:
        blob.upload_from_filename(str(file_path))
    print(f"File {blob_name} uploaded to {bucket_name}.")

def _download_from_bucket(bucket_name: str, blob_name: str, destination_path = None, partial=False):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if partial:
        blob_name = list(bucket.list_blobs(prefix=blob_name))[0].name
    blob = bucket.blob(blob_name)
    if destination_path:
        blob.download_to_filename(str(destination_path))
        return 
    file_content = blob.download_as_bytes()
    return file_content.decode("utf-8")

class CustomRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, *args, llama_tokenize_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.llama_tokenize_function = llama_tokenize_fn
    
    def create_documents(self, texts, metadatas=None):
        documents = super().create_documents(texts, metadatas)
        for i in range(len(documents)):
            llama_tokens = self.llama_tokenize_function(documents[i].page_content)
            documents[i].metadata["llama_tokens"] = llama_tokens
            documents[i].metadata["key"] = documents[i].metadata.get("key") + f"_section_{i}"
        return documents

def get_text_splitter():

    def calculate_llama_token_count(text):
        return calculate_token_count(text, "llama")

    return CustomRecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=0,
        length_function=calculate_llama_token_count,
        llama_tokenize_fn=calculate_llama_token_count
    )