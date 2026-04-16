"""
common.py
---------
Shared utilities used across all data-preparation pipelines.
"""

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
import os
import tiktoken
import time
import unicodedata
import yaml

_TOKENIZERS = {
    "gpt-5": tiktoken.get_encoding("o200k_base"),
    "llama": AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B"),
}

# Load all prompt templates once at module level so every caller shares the same in-memory dict without re-reading the file.
with open(os.path.join("src", "prompts.yaml"), encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

def _get_prompt(prompt: str) -> dict:
    """Retrieve a prompt template by name from the loaded PROMPTS registry.

    Args:
        prompt: Key matching an entry in ``src/prompts.yaml``.

    Returns:
        The prompt dict (typically contains at least a ``"prompt"`` key).
    """
    return PROMPTS[prompt]

def _get_gemini_client() -> genai.Client:
    """Create and return an authenticated Gemini API client via Vertex AI.

    Returns:
        A configured ``genai.Client`` instance ready for API calls.
    """
    return genai.Client(
        vertexai=True,
        project=settings.PROJECT_ID,
        location="global",
    )

def _load_stream(dataset_name: str, config: str, split: str = "train"):
    """Open a HuggingFace dataset as a lazy streaming iterator.

    Args:
        dataset_name: HuggingFace Hub dataset identifier.
        config:       Dataset configuration / subset name.
        split:        Which split to stream (default ``"train"``).

    Returns:
        A ``datasets.IterableDataset`` that yields examples on demand.
    """
    return load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)

def _format_unicode_example(sample: dict) -> dict:
    """Normalise the ``"text"`` field of a dataset example in-place.
    Applies NFKC Unicode normalisation and strips leading/trailing whitespace.
    NFKC converts compatibility characters (e.g. ligatures, full-width forms)
    to their canonical equivalents, which improves tokeniser consistency.

    Args:
        sample: A dataset example dict containing at least a ``"text"`` key.

    Returns:
        The same dict with ``sample["text"]`` normalised.
    """
    sample["text"] = unicodedata.normalize("NFKC", sample["text"].strip())
    return sample

def _add_key_column(dataset: Dataset) -> Dataset:
    """Add a unique ``"key"`` column to a dataset using the zero-based row index.

    Args:
        dataset: A ``datasets.Dataset`` instance.

    Returns:
        A new ``Dataset`` with the ``"key"`` column added.
    """
    return dataset.map(lambda ex, idx: {"key": f"contract_{idx}", **ex}, with_indices=True)

def download_shuffled_samples(
    dataset_name: str,
    config: str,
    n_samples: int,
    general: bool = False,
    seed: int = 42,
) -> Dataset:
    """Stream, shuffle, and collect a fixed number of text samples from a
    HuggingFace dataset.

    Args:
        dataset_name: HuggingFace Hub dataset identifier.
        config:       Dataset configuration / subset name.
        n_samples:    Maximum number of examples to collect.
        general:      If ``True``, skip the jurisdiction filter and collect
                      examples from all jurisdictions.
        seed:         Random seed passed to the shuffle buffer.

    Returns:
        A ``datasets.Dataset`` with a single ``"text"`` column containing
        at most ``n_samples`` Unicode-normalised examples.
    """
    stream = _load_stream(dataset_name, config)
    stream = stream.shuffle(buffer_size=settings.MLP_BUFFER_SIZE, seed=seed)
    collected = []
    for ex in stream:
        if general or ex.get("jurisdiction") == "US":
            # Skip examples where the text is an image path — these originate from scanned contracts that were not OCR-processed correctly.
            if ".jpg" in ex.get("text", ""):
                continue
            collected.append(_format_unicode_example(ex))
            if len(collected) >= n_samples:
                break
    dataset = Dataset.from_list(collected)
    # Retain only the text column to keep the dataset schema minimal.
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    return dataset

def calculate_token_count(text: str, model_name: str) -> int:
    """Count the number of tokens in ``text`` using the specified model's tokeniser.

    Args:
        text:       The input string to tokenise.
        model_name: Key into ``_TOKENIZERS`` — either ``"gpt-5"`` or
                    ``"llama"``.

    Returns:
        The integer token count.
    """
    enc = _TOKENIZERS.get(model_name)
    tokens = enc.encode(text)
    return len(tokens)

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Serialise a list of dicts to a JSONL file (one JSON object per line).

    Args:
        path:    Destination file path. Parent directories must already exist.
        entries: List of dicts to serialise.
    """
    print(f"Writing batch input to {path}")
    with open(path, "w", encoding="utf-8") as output:
        for entry in entries:
            output.write(json.dumps(entry) + "\n")
    print(f"Saved batch input to {path}")

async def create_and_submit_batch_job(
    client,
    entries: list[dict],
    save_batch_path: Path,
    batch_file_name: str,
    batch_job_name: str,
    submit: bool,
):
    """Write a JSONL batch file, upload it to GCS, and optionally create a
    Gemini batch inference job.

    Args:
        client:           A configured Gemini API client.
        entries:          List of request dicts conforming to the Gemini batch
                          input schema.
        save_batch_path:  Local path where the JSONL file should be saved.
        batch_file_name:  Filename component used inside the GCS blob path.
        batch_job_name:   Human-readable display name for the batch job.
        submit:           If ``True``, create the batch job on the Gemini API;
                          if ``False``, upload the file but skip batch job creation.

    Returns:
        The created batch ``Job`` object, or ``None`` if ``submit=False``.
    """
    save_batch_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_write_jsonl, save_batch_path, entries)

    # Timestamp-prefix the blob name to avoid collisions between pipeline runs.
    blob_name = f"{datetime.now().strftime('%Y-%m-%d::%H:%M:%S')}/{batch_file_name}"

    await asyncio.to_thread(
        _upload_to_bucket,
        bucket_name=settings.INS_FT_MLP_BUCKET_NAME,
        blob_name=blob_name,
        jsonL=True,
        jsonL_string="\n".join(json.dumps(entry) for entry in entries),
    )
    print(f"Uploaded to bucket: {settings.INS_FT_MLP_BUCKET_NAME}/{blob_name}")

    if submit:
        job = client.batches.create(
            model="gemini-3-flash-preview",
            src=f"gs://{settings.INS_FT_MLP_BUCKET_NAME}/{blob_name}",
            config=types.CreateBatchJobConfig(
                display_name=batch_job_name,
                dest=f"gs://{settings.INS_FT_MLP_BUCKET_NAME}/{settings.INS_FT_MLP_BATCH_RESULTS}/",
            ),
        )
        print(f"Created batch job {job.name}")
        return job
    else:
        print(f"Batch job creation skipped for: {settings.INS_FT_MLP_BUCKET_NAME}/{blob_name}")
        return None

async def poll_and_store_batch_results(
    job_name: str,
    poll_interval: int = 30,
    timeout: float | None = None,
) -> tuple[list[dict], list[str]]:
    """Poll a Gemini batch job until it reaches a terminal state, then download
    and parse the inference results.

    Args:
        job_name:      The Gemini batch job name
        poll_interval: Seconds to wait between state-check requests (default 30).
        timeout:       Optional wall-clock timeout in seconds. 

    Returns:
        A ``(lines, unable_to_parse)`` tuple where ``lines`` is a list of
        successfully parsed result dicts and ``unable_to_parse`` is a list of
        raw strings that could not be decoded as JSON.
    """
    client = _get_gemini_client()
    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    start = time.time()
    batch_job = client.batches.get(name=job_name)
    while batch_job.state.name not in completed_states:
        if timeout and time.time() - start > timeout:
            raise TimeoutError("Batch job polling exceeded timeout")
        await asyncio.sleep(poll_interval)
        batch_job = await asyncio.to_thread(client.batches.get, name=job_name)
        print(f"Batch job {job_name} state={batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job finished with state {batch_job.state.name}")

    # Use a timestamp-based prefix to locate the correct output blob in GCS.
    decoded_content = _download_from_bucket(
        bucket_name=settings.INS_FT_MLP_BUCKET_NAME,
        blob_name=rf"batch_results/prediction-model-{batch_job.create_time.strftime('%Y-%m-%dT%H:%M')}",
        partial=True,
    )

    # Parse lines individually so a single malformed entry does not discard the entire result batch.
    lines = []
    unable_to_parse = []
    for line in decoded_content.splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            unable_to_parse.append(line)
    return lines, unable_to_parse


def _upload_to_bucket(
    bucket_name: str,
    blob_name: str,
    file_path=None,
    jsonL: bool = False,
    jsonL_string: str | None = None,
) -> None:
    """Upload a file or a JSONL string to a Google Cloud Storage bucket.
    
    Args:
        bucket_name:  Name of the target GCS bucket.
        blob_name:    Destination object name within the bucket.
        file_path:    Local path of the file to upload. Used when ``jsonL=False``.
        jsonL:        If ``True``, upload ``jsonL_string`` as ``application/x-jsonlines`` instead of a local file.
        jsonL_string: JSONL-formatted string to upload. Used when ``jsonL=True``.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if jsonL:
        blob.upload_from_string(jsonL_string, content_type="application/x-jsonlines")
    else:
        blob.upload_from_filename(str(file_path))
    print(f"File {blob_name} uploaded to {bucket_name}.")

def _download_from_bucket(
    bucket_name: str,
    blob_name: str,
    destination_path=None,
    partial: bool = False,
) -> str | None:
    """Download a blob from a Google Cloud Storage bucket.

    Args:
        bucket_name:      Name of the source GCS bucket.
        blob_name:        Object name to download. When ``partial=True`` this is treated as a prefix and the first matching blob is used.
        destination_path: If provided, the blob is saved to this local path and the function returns ``None``.
        partial:          If ``True``, perform a prefix-based blob lookup and use the first match. Useful when the exact blob name is not known in advance (e.g. batch result files).

    Returns:
        The decoded UTF-8 string content when ``destination_path`` is ``None``, otherwise ``None``.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if partial:
        blob_name = list(bucket.list_blobs(prefix=blob_name))[0].name
    blob = bucket.blob(blob_name)
    if destination_path:
        blob.download_to_filename(str(destination_path))
        return None
    file_content = blob.download_as_bytes()
    return file_content.decode("utf-8")

class CustomRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """A ``RecursiveCharacterTextSplitter`` that additionally annotates each
    chunk with its LLaMA token count and a unique section key.
    """

    def __init__(self, *args, llama_tokenize_fn, **kwargs):
        """Initialise the splitter with a LLaMA tokenisation function.

        Args:
            *args:             Positional arguments forwarded to ``RecursiveCharacterTextSplitter``.
            llama_tokenize_fn: Callable ``(text: str) -> int`` that returns the LLaMA token count for a given string.
            **kwargs:          Keyword arguments forwarded to ``RecursiveCharacterTextSplitter``.
        """
        super().__init__(*args, **kwargs)
        self.llama_tokenize_function = llama_tokenize_fn

    def create_documents(self, texts, metadatas=None):
        """Split texts into chunks and annotate each chunk with LLaMA token metadata.

        Args:
            texts:     List of input strings to split.
            metadatas: Optional list of metadata dicts, one per input text.

        Returns:
            List of ``Document`` objects with ``"llama_tokens"`` and unique ``"key"`` fields added to their metadata.
        """
        documents = super().create_documents(texts, metadatas)
        for i, doc in enumerate(documents):
            doc.metadata["llama_tokens"] = self.llama_tokenize_function(doc.page_content)
            doc.metadata["key"] = doc.metadata.get("key") + f"_section_{i}"
        return documents

def get_text_splitter(
        chunk_size: int = 6000,
        chunk_overlap: int = 0,
    ) -> CustomRecursiveCharacterTextSplitter:
    """Construct a pre-configured ``CustomRecursiveCharacterTextSplitter``.

    The same ``calculate_llama_token_count`` closure is passed as both the
    ``length_function`` (used by the parent splitter to measure chunk size)
    and ``llama_tokenize_fn`` (stored on the instance for metadata annotation
    in ``create_documents``).

    Returns:
        A ready-to-use ``CustomRecursiveCharacterTextSplitter`` instance.
    """
    def calculate_llama_token_count(text: str) -> int:
        return calculate_token_count(text, "llama")

    return CustomRecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=calculate_llama_token_count,
        llama_tokenize_fn=calculate_llama_token_count,
    )