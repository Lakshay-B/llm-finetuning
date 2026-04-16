"""
ift.py
------
Instruction fine-tuning (IFT) utilities: model loading, dataset post-processing,
and tokenised-sequence assembly for two training tasks.

Two task types are supported:

  **Summary** — Each contract chunk is paired with its Gemini-generated summary
  to form a supervised ``(user: text, assistant: summary)`` conversation.

  **QA (multi-part clause detection)** — Each contract chunk is paired with
  positive and negative question-answer turns derived from the CUAD entity
  extraction annotations, forming a multi-turn conversation.
"""

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
    """Load the pretrained model and its tokeniser from ``settings.MODEL_PATH``.

    The tokeniser is initialised with the LLaMA-3.1 chat template via Unsloth's
    ``get_chat_template``. The base model weights are loaded in the dtype and
    device configuration specified in the training config, and the LoRA adapter
    stored at ``settings.MODEL_PATH`` is applied via ``PeftModel``.

    Returns:
        A ``(model, tokenizer)`` tuple ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    # Load the training config once to avoid repeated file I/O.
    model_config = load_training_config("MODEL")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        torch_dtype=model_config["dtype"],
        device_map="auto",
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
    """Async wrapper around ``load_model_and_tokenizer``.

    Returns:
        A ``(model, tokenizer)`` tuple, as returned by
        ``load_model_and_tokenizer``.
    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        model, tokenizer = await loop.run_in_executor(executor, load_model_and_tokenizer)
    return model, tokenizer

def process_summary_dataset(sample: dict) -> dict:
    """Unwrap the ``"summary"`` field from its JSON envelope, if present.

    Args:
        sample: A dataset example dict containing at least a ``"summary"`` key.

    Returns:
        The same dict with ``sample["summary"]`` unwrapped.
    """
    try:
        sample["summary"] = json.loads(sample["summary"]).get("summary")
        return sample
    except json.JSONDecodeError:
        return sample

@lru_cache()
def get_process_entity_dataset_resources() -> tuple[dict, list, list]:
    """Build and cache the category lookup structures used by ``process_entity_dataset``.

    Returns:
        A ``(categories_processed, non_bool_categories, bool_categories)`` tuple.
    """
    categories = list(load_cuad_resource("Entities").values())
    non_bool_categories = [
        "Document Name", "Parties", "Agreement Date", "Expiration Date",
        "Effective Date", "Renewal Term", "Notice Period To Terminate Renewal",
        "Governing Law", "Warranty Duration",
    ]
    bool_categories = [cat for cat in categories if cat not in non_bool_categories]
    categories_processed = {cat.replace(" ", ""): cat for cat in categories}
    return categories_processed, non_bool_categories, bool_categories

def process_entity_dataset(entity_dataset: Dataset) -> tuple[Dataset, dict]:
    """Validate and normalise entity extraction results from the Gemini batch job.

    For each sample the ``"entities"`` JSON string is parsed and each entity
    item is checked:

    - If the entity name does not exactly match a known CUAD category it is
      corrected via fuzzy matching (rapidfuzz token-sort-ratio).
    - For boolean categories, common OCR/model artefacts such as ``"n o"`` and
      ``"y e s"`` are normalised to ``"No"`` / ``"Yes"``.
    - Answers that remain outside ``{"yes", "no"}`` after normalisation, as
      well as unrecognised category names, are recorded in ``erroneous_entities``
      for downstream inspection.

    Args:
        entity_dataset: A ``Dataset`` with at least ``"key"``, ``"text"``, ``"entities"`` (JSON string), ``"text_llama_tokens"`` and ``"entities_llama_tokens"`` columns.

    Returns:
        A ``(cleaned_dataset, erroneous_entities)`` tuple where
        ``cleaned_dataset`` has the ``"entities"`` column replaced with a
        list of normalised entity dicts, and ``erroneous_entities`` maps
        problematic sample keys to their error details.
    """
    categories_processed, non_bool_categories, bool_categories = get_process_entity_dataset_resources()
    erroneous_entities = {}
    entity_dataset_list = []

    for sample in entity_dataset:
        try:
            sample_entities = json.loads(sample["entities"])["Entities"]
            sample_entity_list = []

            for entity_item in sample_entities:
                if entity_item["Entity"] not in categories_processed.values():
                    best_match_key = process.extractOne(
                        entity_item["Entity"].replace(" ", ""),
                        categories_processed.keys(),
                        scorer=fuzz.token_sort_ratio,
                    )[0]
                    entity_item["Entity"] = categories_processed[best_match_key]

                if entity_item["Entity"] in bool_categories:
                    # Normalise common model artefacts for boolean answers.
                    answer_norm = entity_item["Answer"].lower().strip()
                    if answer_norm in ["n/a", "n o", "n / a"]:
                        entity_item["Answer"] = "No"
                    elif answer_norm == "y e s":
                        entity_item["Answer"] = "Yes"

                    if entity_item["Answer"].lower().strip() not in ["yes", "no"]:
                        erroneous_entities[sample["key"]] = {
                            "error": "BoolAnswerError",
                            "content": entity_item["Answer"],
                        }
                elif entity_item["Entity"] not in non_bool_categories:
                    # Entity name matched neither a bool nor a non-bool category
                    # — something went wrong with matching.
                    erroneous_entities[sample["key"]] = {
                        "error": "CategoryError",
                        "content": f"{sample['key']}: {entity_item['Entity']}",
                    }

                sample_entity_list.append(entity_item)

            entity_dataset_list.append({
                "key": sample["key"],
                "text": sample["text"],
                "entities": sample_entity_list,
                "text_llama_tokens": sample["text_llama_tokens"],
                "entities_llama_tokens": sample["entities_llama_tokens"],
            })

        except json.JSONDecodeError:
            erroneous_entities[sample["key"]] = {
                "error": "EntitiesJSONDecodeError",
                "content": sample["entities"],
            }

    return Dataset.from_list(entity_dataset_list), erroneous_entities

def multipart_qa_resources() -> tuple[dict, list, list]:
    """Load the resources required to build multi-part QA turn lists.

    Returns:
        A ``(ques_samples, bool_categories, non_bool_categories)`` tuple where
        ``ques_samples`` is a dict mapping each CUAD category label to a list
        of representative question strings.
    """
    _, non_bool_categories, bool_categories = get_process_entity_dataset_resources()
    ques_samples = load_cuad_resource(resource="SampleQuestions")
    return ques_samples, bool_categories, non_bool_categories

def prepare_multipart_qa(
    sample: dict,
    ques_samples: dict,
    bool_cat: list,
    non_bool_cat: list,
) -> dict:
    """Build positive and negative QA turn lists for a single annotated sample.

    For each entity in ``sample["entities"]``:

    - **Positive turns** are created when the answer is substantive (``"Yes"``
      for boolean categories, or any non-N/A value for free-text categories).
      The answer is augmented with the source span as a reference.
    - **Negative turns** are created when the answer is absent/No, capped at 10
      negative turns per sample to avoid class imbalance.

    A random question is sampled from ``ques_samples`` for each entity.

    Args:
        sample: An annotated dataset example with an ``"entities"`` list, each item containing ``"Entity"``, ``"Answer"``, and ``"Span"`` fields.
        ques_samples: Dict mapping CUAD category labels to lists of question strings (from ``load_cuad_resource("SampleQuestions")``).
        bool_cat: List of boolean clause category labels.
        non_bool_cat: List of free-text clause category labels.

    Returns:
        The same ``sample`` dict with a ``"qa_pairs"`` field added:
        ``{"pos": [...], "neg": [...]}``.
    """
    sample["qa_pairs"] = {"pos": [], "neg": []}
    for entity in sample["entities"]:
        if entity["Entity"] in bool_cat:
            if entity["Answer"].lower().strip() == "yes":
                ques = random.choice(ques_samples[entity["Entity"]])
                ans = f"{entity['Answer']}\nReference: {entity['Span']}"
                sample["qa_pairs"]["pos"].append(
                    {"entity": entity["Entity"], "question": ques, "answer": ans}
                )
            else:
                if len(sample["qa_pairs"]["neg"]) <= 10:
                    ques = random.choice(ques_samples[entity["Entity"]])
                    sample["qa_pairs"]["neg"].append(
                        {"entity": entity["Entity"], "question": ques, "answer": "N/A"}
                    )
        elif entity["Entity"] in non_bool_cat:
            if entity["Answer"].lower().strip() not in ["n/a", "no"]:
                ques = random.choice(ques_samples[entity["Entity"]])
                ans = f"{entity['Answer']}\nReference: {entity['Span']}"
                sample["qa_pairs"]["pos"].append(
                    {"entity": entity["Entity"], "question": ques, "answer": ans}
                )
            else:
                if len(sample["qa_pairs"]["neg"]) <= 10:
                    ques = random.choice(ques_samples[entity["Entity"]])
                    sample["qa_pairs"]["neg"].append(
                        {"entity": entity["Entity"], "question": ques, "answer": "N/A"}
                    )
    return sample


def chunk_tokens_ift(batch: dict, tokenizer, task: Literal["summary", "qa"]) -> dict:
    """Apply the chat template to a batch of samples and filter sequences that
    exceed the model's maximum context length.

    For the ``"summary"`` task each sample becomes a three-turn conversation:
    ``(system, user: contract_text, assistant: summary)``.

    For the ``"qa"`` task each sample becomes a multi-turn conversation seeded
    with the contract text, followed by interleaved user/assistant QA turns.
    Positive turns (where the clause is present) are appended first, then
    negative turns (clause absent), stopping as soon as the sequence would
    exceed ``max_seq_length``.

    Args:
        batch:     A batched dict from ``Dataset.map`` with ``"text"`` and either ``"summary"`` (summary task) or ``"qa_pairs"`` (qa task) columns.
        tokenizer: The chat-template-aware tokeniser returned by ``load_model_and_tokenizer``.
        task:      Either ``"summary"`` or ``"qa"``.

    Returns:
        A dict with three equal-length lists:
        - ``"set"``:            The raw message list for each retained sequence.
        - ``"input_ids"``:      Tokenised input IDs.
        - ``"attention_mask"``: All-ones mask matching ``input_ids``.
    """
    all_chunks: dict = {"set": [], "input_ids": [], "attention_mask": []}
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
                tokenize=True,
                add_generation_prompt=False,
            )
            # Skip sequences that exceed the model's context window.
            if len(input_ids) > max_tokens_sequence:
                continue
            all_chunks["set"].append(messages)
            all_chunks["input_ids"].append(input_ids)
            all_chunks["attention_mask"].append([1] * len(input_ids))

    elif task == "qa":
        for text, qa_pairs in zip(batch["text"], batch["qa_pairs"]):
            messages = [
                {"role": "system", "content": _get_prompt("clause_detection_system_prompt")["prompt"]},
                {"role": "user", "content": text},
                {"role": "assistant", "content": "How can I assist you with this contract?"}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )

            for qa_pair in qa_pairs["pos"]:
                messages.append({"role": "user", "content": qa_pair["question"]})
                messages.append({"role": "assistant", "content": qa_pair["answer"]})
                input_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                if len(input_ids) > max_tokens_sequence:
                    break

            # Append negative turns (clause absent) until the sequence is full.
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

            all_chunks["set"].append(messages)
            all_chunks["input_ids"].append(input_ids)
            all_chunks["attention_mask"].append([1] * len(input_ids))

    else:
        raise ValueError("Invalid task type. Must be 'summary' or 'qa'.")

    return all_chunks