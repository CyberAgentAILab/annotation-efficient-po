from typing import List

import torch
from torch import nn
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import llm_blender
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from transformers import AutoModel
import numpy as np
import math


class RewardModel:
    """
    Base class for reward models.
    """

    def __init__(self, reward_model_id):
        pass

    def get_reward(self, question: str, answer: str) -> float:
        pass

    def get_rewards(self, question: str, answers: List[str]) -> List[float]:
        scores = []
        for i in range(len(answers)):
            score = self.get_reward(question, answers[i])
            scores.append(score)
        return scores

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        pass

    def get_pairwise_rewards(self, question: str, answers: List[str]) -> np.ndarray:
        scores_matrix = np.zeros((len(answers), len(answers)))
        for i in range(len(answers)):
            for j in range(len(answers)):
                score = self.get_pairwise_reward(question, answers[i], answers[j])
                scores_matrix[i][j] = score
        return scores_matrix


class OASST(RewardModel):
    """OpenAssistant reward model."""

    def __init__(self, reward_model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_id
        )
        self.reward_model.eval()
        self.reward_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_id)

    def get_reward(self, question, answer):
        # TODO: Batch operation.
        inputs = self.tokenizer(question, answer, return_tensors="pt").to(self.device)
        outputs = self.reward_model(**inputs).logits[0].cpu().detach().numpy().item()
        return outputs


class StanfordNLP(RewardModel):
    """StanfordNLP reward model."""

    def __init__(self, reward_model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(reward_model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(reward_model_id).to(
            self.device
        )
        self.model.eval()

    def get_reward(self, question, answer):
        input_text = "POST: {} \n\n RESPONSE A: {}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE".format(
            question.replace("\n", " "), answer.replace("\n", " ")
        )
        x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
        )
        prob_of_A = torch.exp(outputs.scores[0][:, 71]) / torch.exp(
            outputs.scores[0][:, :]
        ).sum(
            axis=1
        )  # index 71 corresponds to the token for 'A'
        return prob_of_A.cpu().detach().numpy().item()

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        input_text = "POST: {} \n\n RESPONSE A: {}\n\n RESPONSE B: {}\n\n Which response is better? RESPONSE".format(
            question.replace("\n", " "),
            answer.replace("\n", " "),
            compared_answer.replace("\n", " "),
        )
        x = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1
        )
        prob_of_A = torch.exp(outputs.scores[0][:, 71]) / torch.exp(
            outputs.scores[0][:, :]
        ).sum(
            axis=1
        )  # index 71 corresponds to the token for 'A'
        return prob_of_A.cpu().detach().numpy().item()


class Eurus(RewardModel):
    """Eurus reward model."""

    def __init__(self, reward_model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        self.model = AutoModel.from_pretrained(
            reward_model_id, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def get_reward(self, question, answer):
        input_text = "[INST] {} [\INST] {}".format(question, answer)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        chosen_reward = self.model(**inputs).item()
        return chosen_reward


class PairLM(RewardModel):
    """PairLM reward model."""

    def __init__(self, reward_model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")
        self.blender.blender_config.use_tqdm = False

        # This is for pairwise comparison.
        # We don't need self.blender but for backward compatibility, we keep it.
        self.pairrm = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf").to(
            self.device
        )
        self.pairrm.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("llm-blender/PairRM-hf")
        self.source_prefix = "<|source|>"
        self.cand1_prefix = "<|candidate1|>"
        self.cand2_prefix = "<|candidate2|>"

    def get_reward(self, question, answer):
        print("PairLM.get_reward() not implemented.")
        assert False

    def get_rewards(self, question, answers):
        ranks = self.blender.rank(
            [question], [list(answers)], return_scores=False, batch_size=1
        )
        return (1 - ranks[0]).tolist()

    def get_winratio(self, question, answer, compared_answers):
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert isinstance(compared_answers, list)
        assert isinstance(compared_answers[0], str)

        wins = 0
        cs = list(compared_answers)
        ncs = len(cs)
        pairs = [[answer, cs[i]] for i in range(ncs)]
        ranks = self.blender.rank(
            [question] * ncs, pairs, return_scores=False, batch_size=16
        )

        wins = (ranks[:, 0] < ranks[:, 1]).sum() / ncs

        return wins

    def tokenize_pair(
        self,
        sources: List[str],
        candidate1s: List[str],
        candidate2s: List[str],
        source_max_length=1224,
        candidate_max_length=412,
    ):
        ids = []
        assert len(sources) == len(candidate1s) == len(candidate2s)
        max_length = source_max_length + 2 * candidate_max_length
        for i in range(len(sources)):
            source_ids = self.tokenizer.encode(
                self.source_prefix + sources[i],
                max_length=source_max_length,
                truncation=True,
            )
            candidate_max_length = (max_length - len(source_ids)) // 2
            candidate1_ids = self.tokenizer.encode(
                self.cand1_prefix + candidate1s[i],
                max_length=candidate_max_length,
                truncation=True,
            )
            candidate2_ids = self.tokenizer.encode(
                self.cand2_prefix + candidate2s[i],
                max_length=candidate_max_length,
                truncation=True,
            )
            ids.append(source_ids + candidate1_ids + candidate2_ids)
        encodings = self.tokenizer.pad(
            {"input_ids": ids},
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )
        return encodings

    def get_pairwise_reward(
        self, question: str, answer: str, compared_answer: str
    ) -> float:
        encodings = self.tokenize_pair([question], [answer], [compared_answer])
        encodings = {k: v.to(self.pairrm.device) for k, v in encodings.items()}
        outputs = self.pairrm(**encodings)
        logits = outputs.logits.tolist()
        return logits[0]


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores


class Starling(RewardModel):
    def __init__(self, reward_model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("initializing Starling...")
        self.reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf").to(
            self.device
        )
        self.reward_tokenizer = self.reward_model.tokenizer
        self.reward_tokenizer.truncation_side = "left"

        directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        self.reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.reward_model.eval().requires_grad_(False)

    def get_reward_(self, samples):
        """samples: List[str]"""
        input_ids = []
        attention_masks = []
        encodings_dict = self.reward_tokenizer(
            samples,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        input_ids = encodings_dict["input_ids"]
        attention_masks = encodings_dict["attention_mask"]
        mbs = 4
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            rewards = self.reward_model(
                input_ids=input_ids[i * mbs : (i + 1) * mbs],
                attention_mask=attention_masks[i * mbs : (i + 1) * mbs],
            )
            out.extend(rewards)
        return torch.hstack(out)

    def get_reward(self, question, answer):
        sequences = ["<s>[INST] {} </s> [/INST] {}</s>".format(question, answer)]
        return self.get_reward_(sequences).cpu().detach().numpy().item()

    def get_rewards(self, question, answers):
        rewards = []
        for answer in answers:
            rewards.append(self.get_reward(question, answer))
        return rewards


def load_reward_model(reward_model_id: str):
    """
    Currently it only supports the following reward models.
    To add a new reward model, implement it in the RewardModel class.
    """
    if reward_model_id == "OpenAssistant/reward-model-deberta-v3-large-v2":
        return OASST(reward_model_id)
    elif "stanfordnlp/SteamSHP" in reward_model_id:
        return StanfordNLP(reward_model_id)
    elif reward_model_id == "llm-blender/PairRM":
        return PairLM(reward_model_id)
    elif "berkeley-nest" in reward_model_id:
        return Starling(reward_model_id)
    elif "openbmb/Eurus-RM-7b" in reward_model_id:
        return Eurus(reward_model_id)
    else:
        raise ValueError("Invalid reward_model_id")
