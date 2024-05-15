import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from typing import List
from aepo.mbr.utility_func import load_similarity
from aepo.mbr.reward_model import load_reward_model


def run_reward(
    reward_model_id: str,
    instructions: List[str],
    output_dir: str,
    output_filename: str,
    num_instructions: int,
    num_annotations: int,
    west_of_n: bool,
    dmbr_result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Args:
        reward_model_id (str): the Huggingface Hub's repository name of the reward model.
        instructions (List[str]): the list of instructions.
        output_dir (str): the output directory.
        output_filename (str): the output filename.
        num_instructions (int): the number of instructions.
        num_annotations (int): the number of annotations available per instruction.
        west_of_n (bool): if true, output the best and worst of N samples instead of all annotated responses.
        dmbr_result (pd.DataFrame): the diverse MBR results.
    Returns:
        pd.DataFrame: the annotation-efficient preference optimization dataset.
    Annotate the reward for the responses selected by diverse MBR.
    """

    reward_model = load_reward_model(reward_model_id)

    os.makedirs(output_dir, exist_ok=True)

    # filtered_files = sorted(os.listdir(sample_dir))

    rows = []

    for sample_id in tqdm(range(num_instructions)):
        input_instruction = instructions[sample_id]

        dmbr_samples = dmbr_result.iloc[sample_id]

        responses = [dmbr_samples["text_{}".format(i)] for i in range(num_annotations)]
        rewards = reward_model.get_rewards(input_instruction, responses)

        rank = np.argsort(np.array(rewards))

        # print('rank=', rank)

        ordered_responses = [responses[r] for r in rank]
        ordered_responses.reverse()

        ordered_rewards = [rewards[r] for r in rank]
        ordered_rewards.reverse()

        if west_of_n:
            row = (
                [input_instruction]
                + [ordered_responses[0], ordered_responses[-1]]
                + [ordered_rewards[0], ordered_rewards[-1]]
            )
        else:
            row = [input_instruction] + ordered_responses + ordered_rewards

        rows.append(row)

    if (num_annotations == 2) or west_of_n:
        columns = ["prompt", "chosen", "rejected", "chosen_r", "rejected_r"]
    else:
        columns = (
            ["prompt"]
            + ["pref_{}".format(i) for i in range(num_annotations)]
            + ["pref_{}_r".format(i) for i in range(num_annotations)]
        )

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(os.path.join(output_dir, output_filename), index=False)
    return df
