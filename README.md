## Annotation-Efficient Preference Optimization

This repository implements the Annotation-Efficient Preference Optimization (AEPO) algorithm.


## Install

You can install aepo via pip.
```
pip install -i https://test.pypi.org/simple/ aepo
```

Source install is available too.
```
git clone git@github.com:CyberAgentAILab/aepo.git
cd aepo
pip install .
```


## Usage

The command line interface is available.
The input dataset can be csv file or a dataset uploaded to Huggingface Hub.
The dataset should have a column named *prompt* or *instruction*. aepo recognize it as the user prompt given to the system and the rest of the columns to be the responses generated by the system.

I prepared an example dataset in `dataset/alpaca_samples.csv`.
The csv file includes 128 responses generated by [HuggingFaceH4/mistral-7b-sft-beta](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) for each instruction of the `alpaca_human_preference` split of [tatsu-lab/alpaca_farm](https://huggingface.co/datasets/tatsu-lab/alpaca_eval).
You can try aepo using this dataset with the following command:

```
aepo dataset/alpaca_samples.csv --num_responses 8 --num_annotations 2 --num_instructions 10
```

`--num_responses` is the number of input responses you use. The dataset has to have responses larger than or equal to `--num_responses`. `--num_annotations` is the number of responses after the subsampling process. It is also the number of times the reward model is queried per instruction.

### Example: Running AEPO

You can generate a pair of responses for each instruction using aepo using the following command.

```
aepo dataset/alpaca_samples.csv --num_responses 8 --num_annotations 2 --num_instructions 10
```

If you want to subsample four responses for e.g, [LiPO](https://arxiv.org/abs/2402.01878v1), then set `--num_annotations` to four.

```
aepo dataset/alpaca_samples.csv --num_responses 8 --num_annotations 4 --num_instructions 10
```

### Example: Running West-of-N over 8 samples

You can run West-of-N using this package simply by setting `--num_annotations` == `--num_responses`.

```
aepo dataset/alpaca_samples.csv --num_responses 8 --num_annotations 8 --num_instructions 10
```

This command will generate a dataset with 8 responses, ranked by the reward. If you only need the best and worst of the N samples, then use `--west_of_n` option.

```
aepo dataset/alpaca_samples.csv --num_responses 8 --num_annotations 8 --num_instructions 10 --west_of_n
```

This will pick the best and worst of the responses.