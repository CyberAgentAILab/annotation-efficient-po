import os
import sys
import logging
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset
import datasets


def read_dataset(
    file_path: str, split: str, access_token: str = None
) -> datasets.Dataset:
    """
    Args:
        file_path (str): the path or the repository name in Huggingface hub of the input dataset file.
        split (str): the split of the dataset to use.
        access_token (str): the read access token for the Huggingface API.
    Returns:
        datasets.Dataset: the annotation-efficient dataset.
    Read the dataset from a file or Huggingface Hub.
    """
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
            "Error: Could not load a file or find a repository in Huggingface hub: " + file_path
        )
        sys.exit("Error: Could not load a file or find a repository in Huggingface hub: " + file_path)

    if not (("prompt" in dataset.features.keys()) or ("instruction" in dataset.features.keys())):
        logging.error(
            "Error: The dataset does not have 'prompt' or 'instruction' field. aepo requires this field to generate the preference dataset."
        )
        sys.exit("Error: The dataset does not have 'prompt' or 'instruction' field. aepo requires this field to generate the preference dataset.")

    return dataset

def ds2csv(
    ds: datasets.Dataset,
    sample_dir: str,
    num_instructions: int = 4,
    num_responses: int = 32,
):
    """
    Convert the dataset to CSV files.
    Args:
        ds (datasets.Dataset): the annotation-efficient dataset.
        sample_dir (str): the directory to save the CSV files.
        num_instructions (int): the number of instructions.
        num_responses (int): the number of responses per instruction we use for the AEPO.
    Returns:
        None
    """
    os.makedirs(sample_dir, exist_ok=True)
    keys = [k for k in ds.features.keys() if ((k != "prompt") or (k != "instruction"))]
    keys = keys[:num_responses]

    for i in tqdm(range(min(num_instructions, len(ds)))):

        rows = []
        for k in keys:
            rows.append([ds[i][k], None])
        df = pd.DataFrame(rows, columns=["text", "probability"])
        df.to_csv("{}/{:06d}.csv".format(sample_dir, i), index=False)
