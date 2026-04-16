"""
cuad_data.py
------------
Utilities for loading, processing, and synthesizing data from the CUAD
(Contract Understanding Atticus Dataset).

This module provides:
  - ``load_cuad_resource``: Cached loader for all CUAD resource files
    (question samples, entities mapping, category descriptions, master data,
    and the dataset README).
  - ``get_master_cuad``: Augments the master clause CSV with fuzzy-matched
    reference filenames and derives the category-to-entity mapping.
  - ``synthesize_cuad_ques_resources`` / ``synthesize_cuad_ques``: Calls a
    Gemini model API to generate representative questions for each clause category.
  - ``process_row`` / ``create_cuad_validation_samples``: Assembles per-contract
    validation JSON files pairing randomly sampled questions with ground-truth
    answers from the master clause annotations.
"""

from google.genai import types
from functools import lru_cache
from pathlib import Path
from pydantic import create_model
from rapidfuzz import process, fuzz
from src.core.settings import settings
from src.get_training_data.common import _get_prompt, _get_gemini_client
from typing import List, Literal

import pandas as pd
import os
import json
import random
import re

@lru_cache()
def load_cuad_resource(
    resource: Literal[
        "SampleQuestions",
        "Entities",
        "ClauseCategoryDescriptions",
        "MasterData",
        "CUADReadme",
    ]
):
    """Load a named CUAD resource from disk.

    Results are memoised via ``@lru_cache`` so repeated calls for the same
    resource incur no additional I/O.

    Args:
        resource: The resource to load. Accepted values:

            - ``"SampleQuestions"``: Dict mapping clause category names to lists
              of representative questions.
            - ``"Entities"``: Dict mapping sanitised column-name keys to their
              human-readable clause category labels.
            - ``"ClauseCategoryDescriptions"``: Dict of per-category textual
              descriptions used in prompts.
            - ``"MasterData"``: The raw ``master_clauses.csv`` as a
              ``pandas.DataFrame``.
            - ``"CUADReadme"``: Full text of ``CUAD_v1_README.txt``.

    Returns:
        The loaded resource in its native Python type (dict, DataFrame, or str).
    """
    if resource == "SampleQuestions":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_QUES_SAMPLES, "r", encoding="utf-8") as f:
                ques_samples = json.load(f)
            return ques_samples
        except FileNotFoundError:
            # File not yet generated — synthesise via the Gemini model and retry.
            try:
                print("Question samples not found. Attempting to synthesize.")
                synthesize_cuad_ques()
                with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_QUES_SAMPLES, "r", encoding="utf-8") as f:
                    ques_samples = json.load(f)
                return ques_samples
            except Exception:
                print("Unable to load question samples.")
                raise

    elif resource == "Entities":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_ENTITIES, "r", encoding="utf-8") as f:
                entities = json.load(f)
            return entities
        except FileNotFoundError:
            # Derive the entities mapping from the master CSV and persist it.
            try:
                get_master_cuad()
                with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_ENTITIES, "r", encoding="utf-8") as f:
                    entities = json.load(f)
                return entities
            except Exception:
                print("Unable to load Entities file.")
                raise

    elif resource == "ClauseCategoryDescriptions":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_CATEGORY_DESCRIPTION, "r", encoding="utf-8") as f:
                clause_category_descriptions = json.load(f)
            return clause_category_descriptions
        except FileNotFoundError:
            print("Clause category descriptions file not found. Ensure that file exists.")
            raise

    elif resource == "MasterData":
        try:
            master_cuad_data = pd.read_csv(settings.INS_FT_CUAD_DATA_DIR / "master_clauses.csv")
            return master_cuad_data
        except FileNotFoundError:
            print("Master CUAD data file not found. Ensure that file exists.")
            raise

    elif resource == "CUADReadme":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / "CUAD_v1_README.txt", "r", encoding="utf-8") as f:
                readme = f.read()
            return readme
        except FileNotFoundError:
            print("CUAD README file not found. Ensure that file exists.")
            raise

    else:
        raise ValueError(
            "Invalid resource type requested. Choose from 'SampleQuestions', 'Entities', 'ClauseCategoryDescriptions', 'MasterData', or 'CUADReadme'."
        )


def get_master_cuad() -> pd.DataFrame:
    """Load the master CUAD clause CSV, enrich it with matched filenames, and
    persist the clause-category-to-entity mapping to disk.

    The function also writes a JSON file (path configured by
    ``settings.INS_FT_CUAD_ENTITIES``) that maps sanitised column-name keys
    to their original human-readable clause category labels. This file is
    consumed by ``load_cuad_resource("Entities")``.

    Returns:
        The enriched master clause DataFrame.
    """
    try:
        master_cuad_data = load_cuad_resource("MasterData")

        # Fuzzy-match each PDF-derived filename to the actual .txt file on disk to account for minor naming differences between the two sources.
        txt_files = os.listdir(settings.INS_FT_CUAD_DATA_DIR / Path("full_contract_txt"))
        master_cuad_data["RefFilename"] = master_cuad_data["Filename"].map(
            lambda x: process.extractOne(x, txt_files, scorer=fuzz.token_sort_ratio)[0]
        )

        # Normalise filenames to a lowercase, extension-free form for use as stable dictionary keys and output filenames.
        master_cuad_data["FilenameProcessed"] = master_cuad_data["Filename"].map(
            lambda x: ".".join(x.lower().split(".")[:-1]).strip()
        )
        master_cuad_data["RefFilenameProcessed"] = master_cuad_data["RefFilename"].map(
            lambda x: ".".join(x.lower().split(".")[:-1]).strip()
        )
        print("Added processed filenames in CUAD master data.")

        # Identify clause-category columns: exclude answer columns and all filename/metadata columns added above.
        interpretability_cols = [
            col for col in master_cuad_data.columns
            if "Answer" not in col
            and col not in ["Filename", "RefFilenameProcessed", "FilenameProcessed", "RefFilename"]
        ]

        ques_cat_mapping = {
            re.sub(r"\\|\/", "_", re.sub(r"[\s-]", "", col)): col
            for col in interpretability_cols
        }

        with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_ENTITIES, "w", encoding="utf-8") as f:
            json.dump(ques_cat_mapping, f)
        print("Saved category-entity mapping for CUAD categories.")
        return master_cuad_data

    except Exception as e:
        print(f"An error occurred while processing master CUAD data and saving question samples: {e}\nCannot proceed with testing.")
        raise

def synthesize_cuad_ques_resources() -> tuple[str, object]:
    """Prepare the prompt and Gemini client needed for question synthesis.

    Returns:
        A ``(prompt_text, client)`` tuple where ``prompt_text`` is the
        fully-rendered prompt string and ``client`` is a configured Gemini
        API client ready for ``generate_content`` calls.
    """
    readme = load_cuad_resource("CUADReadme")

    # The README is delimited by long equal-sign dividers; extract only the sections that contain category and task descriptions.
    cat_and_tasks_sections = "\n".join(
        section.strip()
        for section in readme.split("=================================================")
        if any(keyword in section for keyword in ["CATEGORIES AND TASKS", "CATEGORY LIST"])
    )

    ques_synthesization_prompt = (
        _get_prompt("contract_clause_question_generation")["prompt"]
        .replace("{{README_TEXT}}", cat_and_tasks_sections)
    )
    client = _get_gemini_client()

    return ques_synthesization_prompt, client

def synthesize_cuad_ques() -> None:
    """Generate representative questions for each CUAD clause category using a Gemini model and write them to disk to the path defined by
    ``settings.INS_FT_CUAD_QUES_SAMPLES``, with the sanitised category keys translated back to their original human-readable labels.
    """
    try:
        ques_cat_mapping = load_cuad_resource("Entities")

        # Build a dynamic Pydantic model where each field corresponds to a sanitised category key and holds a list of question strings. 
        # This lets the Gemini API enforce a well-structured JSON response.
        CategoryQuestionsMapping = create_model(
            "ContractResponse",
            **{field: (List[str], ...) for field in ques_cat_mapping.keys()}
        )

        ques_synthesization_prompt, client = synthesize_cuad_ques_resources()

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=ques_synthesization_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CategoryQuestionsMapping,
            ),
        )

        ques_res = json.loads(response.text)
        ques_res = {ques_cat_mapping[k]: v for k, v in ques_res.items()}

        with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_QUES_SAMPLES, "w", encoding="utf-8") as f:
            json.dump(ques_res, f, indent=4)
        print("Saved question samples for CUAD categories.")

    except Exception as e:
        print(f"An error occurred while synthesizing CUAD question samples: {e}\nCannot proceed with testing.")
        raise

def process_row(row: pd.Series) -> dict:
    """Build a validation sample dict for a single contract row.

    For each clause category one question is randomly sampled from the
    pre-synthesised question list and paired with the ground-truth answer and
    interpretation from the master clause annotations.

    Args:
        row: A row from the enriched master CUAD DataFrame (as returned by
            ``get_master_cuad``). Must contain ``RefFilename``, a
            ``<category>-Answer`` column, and a ``<category>`` interpretation
            column for every category present in the question samples.

    Returns:
        A dict with the structure::

            {
                "contract_path": "<absolute path to .txt file>",
                "qa_pairs": [
                    {
                        "category": "<clause category label>",
                        "question": "<sampled question string>",
                        "answer": "<ground-truth answer or 'N/A'>",
                        "interpretation": "<annotator interpretation>",
                    },
                    ...
                ]
            }
    """
    ques_samples = load_cuad_resource("SampleQuestions")

    val_sample = {
        "contract_path": os.path.join(
            settings.INS_FT_CUAD_DATA_DIR, "full_contract_txt", row["RefFilename"]
        ),
        "qa_pairs": [],
    }

    for cat, questions in ques_samples.items():
        # Pick one representative question at random for this category.
        q = random.choice(questions)
        val_sample["qa_pairs"].append({
            "category": cat,
            "question": q,
            "answer": row[f"{cat}-Answer"] if pd.notna(row[f"{cat}-Answer"]) else "N/A",
            "interpretation": row[cat],
        })

    return val_sample


def create_cuad_validation_samples(master_cuad: pd.DataFrame) -> None:
    """Persist per-contract validation JSON files for the entire CUAD dataset.
    Iterates over every row in the enriched master clause DataFrame, calls
    ``process_row`` to assemble the validation sample, and writes the result
    to ``<INS_FT_CUAD_DATA_DIR>/validation/<FilenameProcessed>.json``.

    Args:
        master_cuad: The enriched master clause DataFrame produced by
            ``get_master_cuad``.  Must contain ``FilenameProcessed`` and all
            columns consumed by ``process_row``.
    """
    try:
        for _, row in master_cuad.iterrows():
            val_sample = process_row(row)
            sample_path = os.path.join(
                settings.INS_FT_CUAD_DATA_DIR,
                "validation",
                f"{row['FilenameProcessed']}.json",
            )
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(val_sample, f)
        print("Created validation samples for CUAD data.")
    except Exception as e:
        print(f"An error occurred while creating CUAD validation samples: {e}\nCannot proceed with testing.")
        raise