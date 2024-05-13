import os
import argparse
import logging

from aepo.preprocess import read_dataset, ds2csv
from aepo.mbr.mbr_engine import run_mbr
from aepo.mbr.reward_engine import run_reward


def aepo():
    """The main function for the AEPO command line interface."""
    parser = argparse.ArgumentParser(
        description="AEPO: A Python package for Annotation Efficient Preference Optimization."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="the path or the repository name in Huggingface hub of the input dataset file.",
    )
    parser.add_argument(
        "--cache_dir", type=str, help="the directory to cache the dataset.", default="."
    )
    parser.add_argument(
        "--split", type=str, help="the split of the dataset to use.", default="train"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="the path of the output file.",
        default="preference.csv",
    )
    parser.add_argument(
        "--num_instructions", type=int, help="the number of instructions.", default=4
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        help="(N) the maximum number of responses per instruction in the dataset.",
        default=32,
    )
    parser.add_argument(
        "--num_annotations",
        type=int,
        help="(k) the number of annotations available per instruction. To generate a pairwise preference dataset, set to 2.",
        default=2,
    )
    parser.add_argument(
        "--similarity_measure",
        type=str,
        help="the similarity measure to use for diverse MBR.",
        default="sentbert",
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        help="the diversity penalty for diverse MBR.",
        default=1.0,
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        help="the repository name in Huggingface hub of the reward model. Default is OpenAssistant/reward-model-deberta-v3-large-v2",
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
    )
    parser.add_argument(
        "--west_of_n",
        action="store_true",
        help="use the west-of-n strategy to generate the preference dataset.",
    )
    parser.add_argument(
        "--access_token",
        type=str,
        help="the read access token for the Huggingface API.",
        default=None,
    )
    parser.add_argument(
        "--use_sample_cache", action="store_true", help="use the cached sample dataset."
    )
    parser.add_argument(
        "--use_matrix_cache",
        action="store_true",
        help="use the cached similarity matrix.",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode.")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("1. Preparing the sample dataset...")
    ds = read_dataset(args.dataset, args.split, access_token=args.access_token)
    if args.use_sample_cache:
        logging.info(
            "Using cached sample dataset in "
            + os.path.join(args.cache_dir, "samples")
            + " to skip the preprocessing step."
        )
    else:
        ds2csv(
            ds,
            sample_dir=os.path.join(args.cache_dir, "samples"),
            num_instructions=args.num_instructions,
            num_responses=args.num_responses,
        )
    logging.info("done!")

    logging.info(
        f"2. Subsampling {args.num_annotations} responses out of {args.num_responses} using DMBR..."
    )
    dmbr_result = run_mbr(
        sample_dir=os.path.join(args.cache_dir, "samples"),
        matrix_dir=os.path.join(args.cache_dir, "matrix"),
        dmbr_dir=os.path.join(args.cache_dir, "dmbr"),
        num_instructions=args.num_instructions,
        num_responses=args.num_responses,
        sim=args.similarity_measure,
        use_matrix_cache=args.use_matrix_cache,
        diverse_k=args.num_annotations,
        diversity_penalty=args.diversity_penalty,
    )
    logging.info("done!")

    if "instruction" in ds.features:
        instructions = ds["instruction"]
    elif "prompt" in ds.features:
        instructions = ds["prompt"]
    else:
        assert False, "The dataset does not have 'instruction' or 'prompt' field."

    logging.info("3. Annotating the reward...")
    run_reward(
        reward_model_id=args.reward_model,
        instructions=instructions,
        output_dir=os.path.join(args.cache_dir, "output"),
        output_filename=args.output,
        num_instructions=args.num_instructions,
        num_annotations=args.num_annotations,
        west_of_n=args.west_of_n,
        dmbr_result=dmbr_result,
    )
    logging.info("done!")
    logging.info(
        "The output file is saved in "
        + os.path.join(args.cache_dir, "output", args.output)
    )
