"""
pretraining_data.py
-------------------
Pipeline for assembling and saving the continued pre-training (CPT) dataset.

Three domain slices are downloaded in parallel from HuggingFace Hub:
  - English legal contracts (``Multi_Legal_Pile / en_contracts``)
  - English caselaw        (``Multi_Legal_Pile / en_caselaw``)
  - General web text       (``fineweb-edu``)
"""

from .common import download_shuffled_samples
from concurrent.futures import ThreadPoolExecutor
from datasets import DatasetDict, concatenate_datasets
from src.core.settings import settings

import asyncio

PRETRAINING_DATASETS = [
    ("joelniklaus/Multi_Legal_Pile", "en_contracts", settings.CPT_CONTRACT_SAMPLE_SIZE, False),
    ("joelniklaus/Multi_Legal_Pile", "en_caselaw", settings.CPT_CASELAW_SAMPLE_SIZE, False),
    ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-20", settings.CPT_GEN_SAMPLE_SIZE, True),
]

async def download_multi_datasets_async(seed: int) -> DatasetDict:
    """Download all pretraining dataset slices concurrently and return them as
    a ``DatasetDict``.

    Args:
        seed: Random seed forwarded to each ``download_shuffled_samples`` call to ensure reproducible shuffling across all slices.

    Returns:
        A ``DatasetDict`` with three keys: ``"contracts"``, ``"caselaw"``, and ``"general"`` — each holding a single-column ``Dataset`` of
        Unicode-normalised text examples.
    """
    loop = asyncio.get_running_loop()
    # Run all downloads in parallel using a thread pool sized to the number of dataset slices so every download starts immediately.
    with ThreadPoolExecutor(max_workers=len(PRETRAINING_DATASETS)) as executor:
        tasks = [
            loop.run_in_executor(
                executor, download_shuffled_samples, name, config, n_samples, general, seed
            )
            for name, config, n_samples, general in PRETRAINING_DATASETS
        ]
        contracts_ds, caselaw_ds, general_ds = await asyncio.gather(*tasks)

    return DatasetDict({
        "contracts": contracts_ds,
        "caselaw": caselaw_ds,
        "general": general_ds,
    })

def split_and_save_cpt_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    """Split each dataset slice into train/validation partitions and save the combined CPT dataset to disk.

    Args:
        dataset_dict: A ``DatasetDict`` as returned by ``download_multi_datasets_async``, with keys ``"contracts"``, ``"caselaw"``, and ``"general"``.

    Returns:
        The assembled ``DatasetDict`` with ``"train"`` and ``"val"`` splits.
    """
    train_ds = []
    val_ds = []
    for dataset in dataset_dict.values():
        # Compute the split index once to avoid redundant arithmetic.
        n_train = int(len(dataset) * settings.CPT_TRAIN_RATIO)
        train_ds.append(dataset.take(n_train))
        val_ds.append(dataset.skip(n_train))

    cpt_dataset = DatasetDict({
        "train": concatenate_datasets(train_ds),
        "val": concatenate_datasets(val_ds)
    })
    print(f"Saving pretraining dataset to: {settings.CPT_DATA_DIR}")
    cpt_dataset.save_to_disk(settings.CPT_DATA_DIR)
    return cpt_dataset