import os
from tqdm import tqdm
import gc

import numpy as np
import pandas as pd
import torch

from aepo.mbr.utility_func import load_similarity
from aepo.mbr.policy.mbr import compute_score_matrix
from aepo.mbr.policy.diverse_mbr import compute_dmbr


def run_mbr(
    sample_dir: str,
    matrix_dir: str,
    dmbr_dir: str,
    num_instructions: str,
    num_responses: str,
    sim: str,
    use_matrix_cache: bool,
    diverse_k: int,
    diversity_penalty: float,
) -> pd.DataFrame:
    """
    Args:
        sample_dir (str): the directory of the samples.
        matrix_dir (str): the directory of the similarity matrices.
        dmbr_dir (str): the directory of the diverse MBR results.
        num_instructions (int): the number of instructions.
        num_responses (int): the number of responses per instruction.
        sim (str): the similarity model to use.
        use_matrix_cache (bool): the flag to use the matrix cache.
        diverse_k (int): the number responses to select by diverse MBR.
        diversity_penalty (float): the diversity penalty lambda.
    Returns:
        pd.DataFrame: the diverse MBR results.
    Run the diverse MBR algorithm.
    See https://github.com/CyberAgentAILab/diverse-mbr for more details of diverse MBR.
    """
    compute_similarity, sim_model = load_similarity(sim)

    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(dmbr_dir, exist_ok=True)

    filtered_files = sorted(os.listdir(sample_dir))

    assert len(filtered_files) > 0

    rows = []

    for sample_id, filename in tqdm(enumerate(filtered_files)):

        if sample_id >= num_instructions:
            break

        df = pd.read_csv(os.path.join(sample_dir, filename))

        df = df[:num_responses]
        hyp = df.iloc[:]["text"]

        matrix_filename = "{:06d}.npy".format(sample_id)
        matrix = None
        if use_matrix_cache:
            try:
                matrix = np.loadtxt(os.path.join(matrix_dir, matrix_filename))
            except:
                matrix = None
        if matrix is None:
            matrix = compute_score_matrix(hyp, compute_similarity, None)
            matrix_path = os.path.join(matrix_dir, matrix_filename)
            np.savetxt(matrix_path, matrix)

        # Diverse MBR
        dmbr_bests = compute_dmbr(matrix=matrix, k=diverse_k, div_pen=diversity_penalty)
        dmbr_hyps = df["text"].iloc[dmbr_bests].to_list()
        row = [sample_id] + dmbr_bests.tolist() + dmbr_hyps
        rows.append(row)

    columns = (
        ["sample_id"]
        + [f"id_{i}" for i in range(diverse_k)]
        + [f"text_{i}" for i in range(diverse_k)]
    )

    df = pd.DataFrame(rows, columns=columns)

    result_filename = "k{:02d}_lambda{:.3f}.csv".format(diverse_k, diversity_penalty)
    df.to_csv(os.path.join(dmbr_dir, result_filename), index=False)

    # Clean up the similarity model so that it doens't take up too much memory.
    del sim_model
    gc.collect()
    torch.cuda.empty_cache()

    return df
