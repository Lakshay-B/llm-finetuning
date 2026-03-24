from datasets import load_dataset
from datetime import datetime
from google import genai
from google.cloud import storage
from pathlib import Path
from src.core.settings import settings

import unicodedata
import json
import yaml

with open(r"src\prompts.yaml") as f:
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
    return {
        "text": unicodedata.normalize("NFKC", sample["text"].strip())
    }

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    print("Writing batch input to %s", path)
    with open(path, "w", encoding="utf-8") as output:
        for entry in entries:
            output.write(json.dumps(entry) + "\n")
    print("Saved batch input to %s", path)

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