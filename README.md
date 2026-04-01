# Repository Overview

This repository contains resources and tools for fine-tuning a LLama-3.2 model, specifically tailored for legal tasks such as contract summarization, clause detection, and analysis. Below is a detailed breakdown of the repository structure and its contents.

## Repository Structure

### Root Directory
- **.env.example**: Example environment configuration file.
- **pyproject.toml**: Configuration file for Python project dependencies and settings.
- **uv.lock**: Lock file for managing dependencies.
- **finetuning.ipynb**: A Jupyter notebook that orchestrates the training workflow using the source scripts.
- **README.md**: This file, providing an overview of the repository.

### training_data
This directory stores all synthesized training data, including:
- **cpt_data/**: Training and validation data for CPT-based training, in the HF datasets format (directly readable using datasets.load_from_disk).
- **instruction_ft_data/**: Data for instruction fine-tuning.
  - **contracts/**: Instruction fine-tuning data for contracts.
  - **cuad/**: Restructured CUAD dataset for instruction fine-tuning.
  - **summarization/**: Data for summarization tasks.
  - **qna/**: Data for clause detection tasks.

### src
This directory contains all source scripts for data preparation and training:
- **prompts.yaml**: YAML file containing all the prompts used in this finetuning project.
- **get_training_data/**: Scripts for generating training data.
  - **common.py**: Common functions used across data preparation scripts.
  - **cuad_data.py**: Script to restructure the reference CUAD training dataset, and extracting the titles of categories of clauses present in a contract.
  - **instruction_finetuning_data.py**: Script for preparing instruction fine-tuning data.
  - **pretraining_data.py**: Script for preparing pretraining data.
- **core/**: Core modules for the project.
  - **settings.py**: Configuration file specifying paths for data storage, file names and training data parameters.
  - **training/**: Scripts for training workflows.
    - **common.py**: Common training utilities.
    - **cpt.py**: Training script for continous pretraining.
    - **ift.py**: Training script for instruction fine-tuning, post CPT.
    - **training_config.yaml**: Configuration file for training parameters.

## Key Features

1. **Fine-tuning Workflow**: The `finetuning.ipynb` notebook integrates functions from the source scripts to execute the training data synthesis and model training workflow.
2. **Training Data Management**: The `training_data` directory organizes all training data, from reference CUAD datasets to pretraining and synthesized instruction fine-tuning data.
3. **Source Code**: The `src` directory contains modular scripts for data preparation and training, ensuring flexibility and reusability.
4. **Application of Google Vertex AI**: This repository leverages Google Vertex GenAI client for all LLM calls and Batch API for training data synthesis.

## Usage

1. **Environment Setup**: Configure the environment using `.env.example` and install dependencies using `uv`. Clone the repository and run `uv sync` in terminal.
2. **Fine-tuning**: Execute the `finetuning.ipynb` notebook to prepare data and run the training pipeline.
3. **Prompts**: Customize prompts in `src/prompts.yaml` as needed for specific tasks.

## Acknowledgments
- CUAD dataset contributors (https://www.atticusprojectai.org/cuad/#dataset).