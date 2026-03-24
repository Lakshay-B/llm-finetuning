from .common import _get_gemini_client, _get_prompt
from src.core.settings import settings

import json

def get_categories_description():
    prompt = _get_prompt("get_categories_description")["prompt"]
    with open(f"instruction_ft_data\cuad\ques_samples.json", "r", encoding="utf-8") as f:
        ques_samples = json.load(f)
    with open(f"instruction_ft_data\cuad\CUAD_v1_README.txt", "r", encoding="utf-8") as f:
        cuad_readme = f.read()
    cuad_readme = cuad_readme.split("=================================================")
    categories_description_text = [_ for _ in cuad_readme if "CATEGORY LIST" in _][0].strip()
    prompt = prompt.replace("{{CATEGORIES_DESCRIPTION_AND_ANSWER_FORMAT}}", categories_description_text).replace("{{CATEGORIES_LIST}}", str(list(ques_samples.keys())))
    client = _get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    category_description = json.loads(response.text)
    with open(r"instruction_ft_data\multi_legal_pile\category-description.json", "w", encoding="utf-8") as f:
        json.dump(category_description, f, ensure_ascii=False, indent=4)