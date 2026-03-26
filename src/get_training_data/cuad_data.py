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
import json

@lru_cache()
def load_cuad_resource(resource: Literal["SampleQuestions", "Entities", "ClauseCategoryDescriptions", "MasterData", "CUADReadme"]):
    if resource == "SampleQuestions":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_QUES_SAMPLES, "r", encoding="utf-8") as f:
                ques_samples = json.load(f)
            return ques_samples
        except FileNotFoundError:
            print("Question samples not found. Ensure that file exists.")
            return None

    elif resource == "Entities":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_ENTITIES, "r", encoding="utf-8") as f:
                entities = json.load(f)
            return entities
        except FileNotFoundError:
            print("Entities file not found. Ensure that file exists.")
            return None
    
    elif resource == "ClauseCategoryDescriptions":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_CLAUSE_CATEGORY_DESCRIPTIONS, "r", encoding="utf-8") as f:
                clause_category_descriptions = json.load(f)
            return clause_category_descriptions
        except FileNotFoundError:
            print("Clause category descriptions file not found. Ensure that file exists.")
            return None
    
    elif resource == "MasterData":
        try:
            master_cuad_data = pd.read_csv(settings.INS_FT_CUAD_DATA_DIR / "master_clauses.csv")
            return master_cuad_data
        except FileNotFoundError:
            print("Master CUAD data file not found. Ensure that file exists.")
            return None
    
    elif resource == "CUADReadme":
        try:
            with open(settings.INS_FT_CUAD_DATA_DIR / "CUAD_v1_README.txt", "r", encoding = "utf-8") as f:
                readme = f.read()
            return readme
        except FileNotFoundError:
            print("CUAD README file not found. Ensure that file exists.")
            return None

    else:
        raise ValueError("Invalid resource type requested. Choose from 'SampleQuestions', 'Entities', 'ClauseCategoryDescriptions', 'MasterData', or 'CUADReadme'.")


def get_master_cuad():
    try:
        master_cuad_data = load_cuad_resource("MasterData")
        master_cuad_data["RefFilename"] = master_cuad_data["Filename"].map(lambda x: process.extractOne(x, os.listdir(settings.INS_FT_CUAD_DATA_DIR / Path("full_contract_txt")), scorer=fuzz.token_sort_ratio)[0])
        master_cuad_data["FilenameProcessed"] = master_cuad_data["Filename"].map(lambda x: ".".join(x.lower().split(".")[:-1]).strip())
        master_cuad_data["RefFilenameProcessed"] = master_cuad_data["RefFilename"].map(lambda x: ".".join(x.lower().split(".")[:-1]).strip())
        print("Added processed filenames in CUAD master data.")
        interpretability_cols = [
            col for col in list(master_cuad_data.columns)
            if "Answer" not in col and col not in ["Filename", "RefFilenameProcessed", "FilenameProcessed", "RefFilename"]
            ]
        ques_cat_mapping = {re.sub(r"\\|\/", "_", re.sub(r"[\s-]", "", i)) : i for i in interpretability_cols}
        
        with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_ENTITIES, "w", encoding="utf-8") as f:
            json.dump(ques_cat_mapping, f)
        print("Saved category-entity mapping for CUAD categories.")

    except Exception as e:
        print(f"An error occurred while processing master CUAD data and saving question samples: {e}\nCannot proceed with testing.")

    return

def synthesize_cuad_ques():
    readme = load_cuad_resource("CUADReadme")
    cat_and_tasks_sections = "\n".join([_.strip() for _ in readme.split("=================================================") if any(keyword in _ for keyword in ["CATEGORIES AND TASKS", "CATEGORY LIST"])])
    ques_synthesization_prompt = _get_prompt('contract_clause_question_generation')["prompt"].replace("{{README_TEXT}}", cat_and_tasks_sections)
    client = _get_gemini_client()
    
    return ques_synthesization_prompt, client

def synthesize_cuad_ques():
    try:
        ques_cat_mapping = load_cuad_resource("Entities")

        CategoryQuestionsMapping = create_model(
                "ContractResponse",
                **{f: (List[str]) for f in list(ques_cat_mapping.keys())}
            )
        ques_synthesization_prompt, client = synthesize_cuad_ques()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents = ques_synthesization_prompt,
            config=types.GenerateContentConfig(
                response_mime_type = "application/json",
                response_schema=CategoryQuestionsMapping
            )
        )
        ques_res = json.loads(response.text)
        ques_res = {ques_cat_mapping[k]: v for k, v in ques_res.items()}
        with open(settings.INS_FT_CUAD_DATA_DIR / settings.INS_FT_CUAD_QUES_SAMPLES, "w", encoding="utf-8") as f:
            json.dump(ques_res, f, indent=4)
        print("Saved question samples for CUAD categories.")
    
    except Exception as e:
        print(f"An error occurred while processing master CUAD data and saving question samples: {e}\nCannot proceed with testing.")
    
    return

def process_row(row):
    ques_samples = load_cuad_resource("SampleQuestions")
    categories = list(ques_samples.keys())
    questions = [{cat: random.choice(ques_samples[cat])} for cat in categories]

    val_sample = {
        "contract_path": os.path.join(settings.INS_FT_CUAD_DATA_DIR, "full_contract_txt", row["RefFilename"]),
        "qa_pairs": []
    }

    for ques in questions:
        cat, q = next(iter(ques.items()))
        val_sample["qa_pairs"].append({
            "category": cat,
            "question": q,
            "answer": row[f"{cat}-Answer"] if pd.notna(row[f"{cat}-Answer"]) else "N/A",
            "interpretation": row[cat]
        })

    return val_sample


def create_cuad_validation_samples(master_cuad: pd.DataFrame):
    for i in range(len(master_cuad)):
        row = master_cuad.loc[i]
        val_sample = process_row(row)
        sample_path = os.path.join(settings.INS_FT_CUAD_DATA_DIR, "validation", f"{row['FilenameProcessed']}.json")
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        with open(sample_path, "w") as f:
            json.dump(val_sample, f)