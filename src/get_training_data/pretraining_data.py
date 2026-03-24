from .common import download_shuffled_samples, _add_key_column
from src.core.settings import settings
from datasets import DatasetDict, concatenate_datasets
from src.core.settings import settings

import asyncio
from concurrent.futures import ThreadPoolExecutor

PRETRAINING_DATASETS = [
        ("joelniklaus/Multi_Legal_Pile", "en_contracts", settings.CPT_CONTRACT_SAMPLE_SIZE, False),
        ("joelniklaus/Multi_Legal_Pile", "en_caselaw", settings.CPT_CASELAW_SAMPLE_SIZE, False),
        ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-20", settings.CPT_GEN_SAMPLE_SIZE, True),
    ]

async def download_multi_datasets_async(seed) -> DatasetDict:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=len(PRETRAINING_DATASETS)) as executor:
        tasks = [
            loop.run_in_executor(executor, download_shuffled_samples, name, config, n_samples, general, seed)
            for name, config, n_samples, general in PRETRAINING_DATASETS
        ]
        contracts_ds, caselaw_ds, general_ds = await asyncio.gather(*tasks)
    
    return DatasetDict({
        "contracts": contracts_ds,
        "caselaw": caselaw_ds,
        "general": general_ds,
    })

async def split_and_save_cpt_dataset(dataset_dict: DatasetDict):
    train_ds = []
    val_ds = []
    for dataset in dataset_dict.values():
        train_ds.append(dataset.take(int(len(dataset) * settings.CPT_TRAIN_RATIO)))
        val_ds.append(dataset.skip(int(len(dataset) * settings.CPT_TRAIN_RATIO)))
    cpt_dataset = DatasetDict({
        "train": concatenate_datasets(train_ds),
        "val": concatenate_datasets(val_ds)
    })
    print("Saving pretraining dataset to %s", settings.CPT_DATA_DIR)
    cpt_dataset.save_to_disk(settings.CPT_DATA_DIR)
    return cpt_dataset