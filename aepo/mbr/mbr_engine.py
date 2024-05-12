import os
from tqdm import tqdm
import gc

import numpy as np
import pandas as pd
import torch

from aepo.mbr.utility_func import load_similarity
from aepo.mbr.policy.mbr import compute_score_matrix
from aepo.mbr.policy.diverse_mbr import compute_dmbr

def run_mbr(sample_dir: str, matrix_dir: str, dmbr_dir: str, num_instructions: str, num_responses: str, 
            sim: str, use_matrix_cache: bool, diverse_k: int, diversity_penalty: float):
    """Run the diverse MBR algorithm."""
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
        hyp = df.iloc[:]['text']

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
        # TODO: Add an option to report the stats
        dmbr_hyps = df['text'].iloc[dmbr_bests].to_list()
        # dmbr_stats = evaluate_diversity(dmbr_hyps, dmbr_scores, src_input, compute_pairwise)
        row = [sample_id] + dmbr_bests.tolist() + dmbr_hyps # + dmbr_stats
        rows.append(row)

    columns = ['sample_id'] + [f"id_{i}" for i in range(diverse_k)] + [f"text_{i}" for i in range(diverse_k)]
    
    # print('columns: ', columns)
    # print('rows', rows)
    df = pd.DataFrame(rows, columns=columns)
    
    result_filename = "k{:02d}_lambda{:.3f}.csv".format(diverse_k, diversity_penalty)
    df.to_csv(os.path.join(dmbr_dir, result_filename), index=False)

    # Clean up the similarity model so that it doens't take up too much memory.
    del sim_model
    gc.collect()
    torch.cuda.empty_cache()

    return df


# if __name__ == "__main__":
#     """
#     This script is the "main function" of the experiment.
#     """
#     parser = get_mbr_parser()
#     args = parser.parse_args()

#     dataset = args.dataset
#     model_name = args.model

#     sample_dir = args.sample_dir
    
#     n_lines = args.n_lines
#     n_samples = args.n_samples

#     epsilon = args.eps
#     topk = args.topk
#     topp = args.topp

#     sim = args.sim
#     eval_func = args.eval

#     # Algorithm config
#     algorithm = args.algorithm
#     recompute_matrix = args.recompute_matrix

    
#     diverse_k = args.diverse_k
#     diversity_penalty = args.diversity_penalty
#     pairwise_eval = args.pairwise_eval

#     do_sample = args.do_sample > 0
