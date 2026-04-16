"""
instruction_finetuning_data.py
------------------------------
Pipeline for building instruction fine-tuning (IFT) datasets from legal contracts.

Two complementary annotation tasks are supported, both backed by the Gemini
Batch API:

  **Summarization** — Each contract chunk is summarised by the model and the
  result is stored alongside the original text for supervised fine-tuning.

  **Multi-part QnA (entity extraction)** — The model extracts answers for all
  41 CUAD clause categories from each contract, producing structured
  ``(entity, answer, span)`` triples.
"""

from .common import (
    create_and_submit_batch_job,
    download_shuffled_samples,
    _add_key_column,
    poll_and_store_batch_results,
    get_text_splitter,
    _get_gemini_client,
    _get_prompt,
    calculate_token_count,
)
from src.get_training_data.cuad_data import load_cuad_resource
from datasets import Dataset, load_from_disk
from pathlib import Path
from src.core.settings import settings

import asyncio
import json

def split_training_samples(batch: dict) -> dict:
    """Batch-map function that splits raw contract texts into token-bounded chunks
    and filters out chunks that are too short to be useful training examples.
    Intended to be passed directly to ``Dataset.map(..., batched=True)``.

    Args:
        batch: A dict with ``"text"`` and ``"key"`` lists, as produced by the
               HuggingFace batched map interface.

    Returns:
        A dict with three equal-length lists:
        - ``"text"``: filtered chunk content strings.
        - ``"key"``: unique chunk identifiers (``<original_key>_section_<i>``).
        - ``"text_llama_tokens"``: LLaMA token count for each chunk.
    """
    all_texts = []
    all_keys = []
    all_llama_tokens = []
    text_splitter = get_text_splitter()

    for sample_text, sample_key in zip(batch["text"], batch["key"]):
        split_docs = text_splitter.create_documents(
            texts=[sample_text], metadatas=[{"key": sample_key}]
        )
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

def download_contracts_dataset(seed: int, split: bool = True) -> Dataset:
    """Download English legal contracts from Multi_Legal_Pile, assign unique keys,
    optionally chunk them, and save the result to disk.

    Args:
        seed:  Random seed used for shuffling the streaming dataset.
        split: If ``True`` (default), apply ``split_training_samples`` to break
               each contract into token-bounded chunks and filter short ones.
               Set to ``False`` to retain full-length contract texts.

    Returns:
        The processed ``Dataset`` saved to ``settings.INS_FT_DATA_DIR``.
    """
    dataset = download_shuffled_samples(
        dataset_name="joelniklaus/Multi_Legal_Pile",
        config="en_contracts",
        n_samples=settings.INS_FT_MLP_CONTRACT_SAMPLE_SIZE,
        general=False,
        seed=seed,
    )
    dataset = _add_key_column(dataset)
    if split:
        dataset = dataset.map(
            split_training_samples,
            batched=True,
            remove_columns=dataset.column_names,
        )
    print(f"Saving contracts dataset to {settings.INS_FT_DATA_DIR}")
    dataset.save_to_disk(settings.INS_FT_DATA_DIR)
    return dataset

def load_contracts_dataset(limit: int | None = None) -> Dataset:
    """Load the pre-processed contracts dataset from disk.

    Args:
        limit: If provided, only the first ``limit`` examples are returned.

    Returns:
        The ``Dataset`` stored at ``settings.INS_FT_DATA_DIR``, optionally
        truncated to ``limit`` examples.
    """
    if not settings.INS_FT_DATA_DIR.exists():
        raise FileNotFoundError(
            "Keyed contracts dataset not found. Run download_contracts_dataset() first."
        )
    try:
        dataset = load_from_disk(settings.INS_FT_DATA_DIR)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return dataset
    except Exception as e:
        raise RuntimeError(f"Error loading contracts dataset: {e}")


async def parse_results(
    job_name: str,
    text_column_name: str,
    text_column_token_count_name: str,
    batch_results_path: Path,
    key_column: str = "key",
    dataset_limit: int | None = None,
) -> tuple[Dataset, dict]:
    """Poll a Gemini batch job, join the responses back to the contracts dataset,
    and save the annotated dataset to disk.

    For each successful response the model output is stored in
    ``text_column_name`` and its LLaMA token count in
    ``text_column_token_count_name``. Responses that are missing content or
    cannot be parsed as JSON are recorded in the returned ``erroneous_keys``
    dict for downstream inspection.

    Args:
        job_name: Fully-qualified Gemini batch job name.
        text_column_name: Column name to use for the model's text output in the result dataset.
        text_column_token_count_name: Column name for the LLaMA token count of the model output.
        batch_results_path: Directory path where the annotated dataset will be saved.
        key_column: Column used to join batch responses to the contracts dataset (default ``"key"``).
        dataset_limit: If set, only the first ``dataset_limit`` contracts are loaded (useful for testing).

    Returns:
        A ``(results_dataset, erroneous_keys)`` tuple where
        ``results_dataset`` is the contracts dataset with model outputs joined
        in, and ``erroneous_keys`` is a dict mapping problematic contract keys
        to their error details.
    """
    lines, unable_to_parse = await poll_and_store_batch_results(job_name)
    contracts = load_contracts_dataset(limit=dataset_limit)

    records: dict = {}
    erroneous_keys: dict = {}

    for line in lines:
        key = line["key"]
        parts = (
            line.get("response", {})
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
        )
        if parts == {}:
            erroneous_keys[key] = {
                "error": "ErrorInResponse",
                "content": "No parts found in response",
            }
        text = parts.get("text", "N/A")
        token_count = calculate_token_count(text, "llama") if text != "N/A" else 0
        records[key] = {
            text_column_name: text,
            text_column_token_count_name: token_count,
        }
        try:
            json.loads(text)
        except Exception as e:
            erroneous_keys[key] = {"error": "ResponseJSONDecodeError", "content": str(e)}
            continue

    def add_fields(batch: dict) -> dict:
        """Batched map function that joins pre-computed model responses onto the contracts dataset using the ``key_column`` as the join key."""
        response_texts = []
        response_texts_tokens = []
        
        for k in batch["key"]:
            entry = records.get(k)
            if entry:
                response_texts.append(entry[text_column_name])
                response_texts_tokens.append(entry[text_column_token_count_name])
            else:
                # Contract key had no matching batch response (e.g. was skipped).
                response_texts.append("N/A")
                response_texts_tokens.append(0)
        return {
            text_column_name: response_texts,
            text_column_token_count_name: response_texts_tokens,
        }

    results_dataset = contracts.map(add_fields, batched=True)

    batch_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving batch results to {batch_results_path}")
    await asyncio.to_thread(results_dataset.save_to_disk, batch_results_path)
    return results_dataset, erroneous_keys

################################################################ Summarization ################################################################

def get_summarization_resources() -> tuple[str, object, dict]:
    """Load the prompt template, Gemini client, and JSON response schema for the
    contract summarization task.

    Returns:
        A ``(prompt_template, client, schema)`` tuple.
    """
    prompt_template = _get_prompt("contract_summarization")["prompt"]
    client = _get_gemini_client()

    # Minimal schema: the model must produce exactly one "summary" string.
    SUMMARIZATION_RESPONSE_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "summary": {
                "type": "STRING",
                "description": "A clear, structured summary of the contract.",
            }
        },
        "required": ["summary"],
    }

    return prompt_template, client, SUMMARIZATION_RESPONSE_SCHEMA

async def submit_summarization_batch_job(
    dataset: Dataset,
    key_column: str | None = None,
    display_name: str = "MLP_Contract_Summarization_Batch",
    submit: bool = True,
):
    """Build and optionally submit a Gemini batch job for contract summarization.

    Args:
        dataset:      The contracts ``Dataset`` to summarise (must have ``"text"`` and ``key_column`` columns).
        key_column:   Column used as the unique request key (default ``"key"``).
        display_name: Human-readable name for the batch job on the Gemini API.
        submit:       If ``True``, create the job; if ``False``, only write and upload the batch file without submitting.

    Returns:
        The created batch ``Job`` object, or ``None`` if ``submit=False``.
    """
    prompt_template, client, schema = get_summarization_resources()
    entries = []
    key_column = key_column or "key"

    for contract in dataset:
        entries.append({
            "key": contract[key_column],
            "method": "generateContent",
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt_template.replace("{{CONTRACT_TEXT}}", contract["text"])}],
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    "thinkingConfig": {"thinkingLevel": "MINIMAL"}
                },
            },
        })

    job = await create_and_submit_batch_job(
        client=client,
        entries=entries,
        save_batch_path=settings.INS_FT_MLP_DATA_DIR_SUMM / settings.INS_FT_MLP_BATCH_INPUT,
        batch_file_name=str(settings.INS_FT_MLP_BATCH_INPUT),
        batch_job_name=display_name,
        submit=submit,
    )
    return job

################################################################ Multi-part QnA ################################################################

def get_categories_description() -> None:
    """Generate and persist per-category descriptions for all 41 CUAD clause
    categories using a Gemini model.

    This only needs to be called once; subsequently the descriptions are loaded
    via ``load_cuad_resource("ClauseCategoryDescriptions")``.
    """
    try:
        categories_description_prompt = _get_prompt("get_categories_description")["prompt"]
        ques_cat_mapping = load_cuad_resource("Entities")

        cuad_readme_sections = load_cuad_resource("CUADReadme").split(
            "================================================="
        )
        categories_description_text = [
            s for s in cuad_readme_sections if "CATEGORY LIST" in s
        ][0].strip()

        prompt = (
            categories_description_prompt
            .replace("{{CATEGORIES_DESCRIPTION_AND_ANSWER_FORMAT}}", categories_description_text)
            .replace("{{CATEGORIES_LIST}}", str(list(ques_cat_mapping.values())))
        )
        client = _get_gemini_client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        category_description = json.loads(response.text)
        dest = settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_CATEGORY_DESCRIPTION
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(category_description, f, indent=4)
        print(f"Saved category descriptions for CUAD categories: {dest}.")

    except Exception as e:
        print(f"An error occurred while generating category descriptions: {e}")
        raise

def get_qna_resources() -> tuple[str, object, dict]:
    """Load the prompt template, Gemini client, and JSON response schema for the
    multi-part entity extraction (QnA) task.

    Returns:
        A ``(prompt_template, client, schema)`` tuple.
    """
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
                            "description": "The name of the entity being checked.",
                        },
                        "Answer": {
                            "type": "STRING",
                            "description": "The extracted answer or value for the entity in the specified format.",
                        },
                        "Span": {
                            "type": "STRING",
                            "description": "The exact text span from the contract where the entity is found/extracted.",
                        },
                    },
                    "required": ["Entity", "Answer", "Span"],
                },
            }
        },
        "required": ["Entities"],
    }
    return prompt_template, client, QNA_RESPONSE_SCHEMA

async def submit_qna_batch_job(
    dataset: Dataset,
    key_column: str | None = None,
    display_name: str = "MLP_Contract_QnA_Batch",
    submit: bool = True,
):
    """Build and optionally submit a Gemini batch job for multi-part entity
    extraction (QnA) across all 41 CUAD clause categories.

    Args:
        dataset:      The contracts ``Dataset`` to annotate (must have ``"text"`` and ``key_column`` columns).
        key_column:   Column used as the unique request key (default ``"key"``).
        display_name: Human-readable name for the batch job on the Gemini API.
        submit:       If ``True``, create the job; if ``False``, only write and upload the batch file without submitting.

    Returns:
        The created batch ``Job`` object, or ``None`` if ``submit=False``.
    """
    prompt_template, client, schema = get_qna_resources()
    entries = []
    key_column = key_column or "key"

    try:
        category_description = load_cuad_resource("ClauseCategoryDescriptions")
    except:
        get_categories_description()
        category_description = load_cuad_resource("ClauseCategoryDescriptions")

    # Serialise the category descriptions list to a string for prompt injection.
    categories_str = ",\n".join(str(cat) for cat in category_description)

    for contract in dataset:
        entries.append({
            "key": contract[key_column],
            "method": "generateContent",
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": (
                            prompt_template
                            .replace("{{ENTITIES_TO_CHECK}}", categories_str)
                            .replace("{{CONTRACT_TEXT}}", contract["text"])
                        ),
                    }],
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                },
            },
        })

    job = await create_and_submit_batch_job(
        client=client,
        entries=entries,
        save_batch_path=settings.INS_FT_MLP_DATA_DIR_QNA / settings.INS_FT_MLP_BATCH_INPUT,
        batch_file_name=str(settings.INS_FT_MLP_BATCH_INPUT),
        batch_job_name=display_name,
        submit=submit,
    )
    return job