import os
import logging
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset
import datasets


def read_dataset(
    file_path: str, split: str, access_token: str = None
) -> datasets.Dataset:
    """Read the dataset from a file or Huggingface Hub."""
    try:
        if os.path.exists(file_path):
            print("Loading data from file: " + file_path)
            # dataset loaded from csv will automatically be train split.
            dataset = load_dataset("csv", data_files=file_path)["train"]
        else:
            print("Loading data from Huggingface Hub: " + file_path)
            dataset = load_dataset(file_path, token=access_token)[split]
    except Exception as e:
        logging.exception(
            "Error: Could not load data from file or Huggingface hub: " + file_path
        )
        return

    return dataset


def ds2csv(
    ds: datasets.Dataset,
    sample_dir: str,
    num_instructions: int = 4,
    num_responses: int = 32,
):
    os.makedirs(sample_dir, exist_ok=True)
    """Convert the dataset to CSV files."""
    for i in tqdm(range(min(num_instructions, len(ds)))):

        keys = [k for k in ds[i].keys() if (k != "prompt") or (k != "instruction")]

        rows = []
        for k in keys[:num_responses]:
            rows.append([ds[i][k], None])
        df = pd.DataFrame(rows, columns=["text", "probability"])
        df.to_csv("{}/{:06d}.csv".format(sample_dir, i), index=False)
