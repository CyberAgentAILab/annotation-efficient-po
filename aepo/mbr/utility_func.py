import numpy as np

from evaluate import load
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity


def load_similarity(sim: str) -> callable:
    """
    Args:
        sim (str): the name of the similarity function to load.
    Returns:
        callable: the similarity function.
    Load the similarity function to be used for diverse MBR.
    For the purpose of AEPO, we use Sentence BERT (sentbert).
    """
    if sim == "bertscore":
        similarity = load(sim)

        def compute_similarity(hyp, ref, src):
            return similarity.compute(predictions=hyp, references=ref, lang="en")["f1"]

    elif sim == "deberta":
        # This is a better bertscore model. Not sure how much it helps.
        similarity = load("bertscore")

        def compute_similarity(hyp, ref, src):
            return similarity.compute(
                predictions=hyp,
                references=ref,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
            )["f1"]

    elif sim == "sacrebleu":
        similarity = load(sim)

        def compute_similarity(hyp, ref, src):
            scores = [
                similarity.compute(predictions=[hyp[i]], references=[ref[i]])["score"]
                for i in range(len(hyp))
            ]
            return scores

    # elif sim == 'infolm':
    #     similarity = InfoLM('google/bert_uncased_L-2_H-128_A-2',
    #                         information_measure='fisher_rao_distance',
    #                         idf=False, return_sentence_level_score=True)
    #     def compute_similarity(hyp, ref, src):
    #         return -np.array(similarity(hyp, ref)[1])
    # elif sim == 'clip':
    #     # This computes the RefCLIPScore, not the reference-less CLIPScore.
    #     # TODO: there is no similarity function for this
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     model_id = "openai/clip-vit-large-patch14"
    #     # model_id = "openai/clip-vit-base-patch32"
    #     processor = CLIPProcessor.from_pretrained(model_id)

    #     model = CLIPModel.from_pretrained(model_id).to(device)
    #     similarity = CLIPTextModel.from_pretrained(model_id).to(device)
    #     model.eval()
    #     similarity.eval()

    #     def compute_similarity(hyp, ref, src):
    #         with torch.no_grad():
    #             hyp = list(hyp)
    #             ref = list(ref)
    #             inputs = processor(text=hyp + ref, images=src[0], return_tensors="pt", padding="max_length").to('cuda')

    #             text_embeddings = torch.flatten(similarity(inputs.input_ids.to(device))['last_hidden_state'],1,-1)
    #             hyp_embeddings = text_embeddings[:len(hyp)]
    #             ref_embeddings = text_embeddings[len(hyp):]
    #             text_scores = cosine_similarity(hyp_embeddings, ref_embeddings).cpu().detach().numpy()
    #             # print('text_scores.shape=', text_scores.shape)

    #             # Assume the src is the same for all the hypotheses.
    #             # TODO: Reuse the embedding
    #             img_inputs = processor(text=hyp, images=src[0], return_tensors="pt", padding="max_length").to('cuda')
    #             img_outputs = model(**img_inputs)

    #             img_scores = np.squeeze((img_outputs.logits_per_image / 100).cpu().detach().numpy())
    #             # print('img_scores.shape=', img_scores.shape)

    #             harmonic_mean = 2 * text_scores * img_scores / (text_scores + img_scores)
    #         # print('harmonic_mean=', harmonic_mean)
    #         return harmonic_mean
    # elif sim == 'cliptext':
    #     # This computes the RefCLIPScore, not the reference-less CLIPScore.
    #     # TODO: there is no similarity function for this
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     model_id = "openai/clip-vit-large-patch14"
    #     # model_id = "openai/clip-vit-base-patch32"
    #     processor = CLIPProcessor.from_pretrained(model_id)

    #     similarity = CLIPTextModel.from_pretrained(model_id).to(device)
    #     similarity.eval()

    #     def compute_similarity(hyp, ref, src):
    #         with torch.no_grad():
    #             hyp = list(hyp)
    #             ref = list(ref)
    #             inputs = processor(text=hyp + ref, images=src[0], return_tensors="pt", padding="max_length").to('cuda')

    #             text_embeddings = torch.flatten(similarity(inputs.input_ids.to(device))['last_hidden_state'],1,-1)
    #             hyp_embeddings = text_embeddings[:len(hyp)]
    #             ref_embeddings = text_embeddings[len(hyp):]
    #             text_scores = cosine_similarity(hyp_embeddings, ref_embeddings).cpu().detach().numpy()

    #         return text_scores
    # elif sim == 'unigramf1':
    #     similarity = ToktokTokenizer()
    #     def compute_similarity(hyp, ref, src):
    #         nhyp = len(hyp)
    #         f1s = []
    #         for i in range(nhyp):
    #             h = hyp[i]
    #             r = ref[i]
    #             hyp_tok = similarity.tokenize(h)
    #             ref_tok = similarity.tokenize(r)

    #             if len(hyp_tok) == 0 or len(ref_tok) == 0:
    #                 f1s.append(0.0)
    #             else:
    #                 precision = len([token for token in hyp_tok if token in ref_tok]) / len(hyp_tok)
    #                 recall = len([token for token in hyp_tok if token in ref_tok]) / len(ref_tok)

    #                 if precision + recall < 0.0001:
    #                     # Prevent zero division.
    #                     f1s.append(0.0)
    #                 else:
    #                     f1s.append(2.0 * precision * recall / (precision + recall))
    #         return f1s
    elif sim == "sentbert":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        evaluator = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        evaluator.eval()
        # similarity = None

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def compute_similarity(hyp, ref, src):
            hyp = list(hyp)
            ref = list(ref)
            # print('hyp=', hyp)
            # print('ref=', ref)
            with torch.no_grad():
                encoded_input = tokenizer(
                    hyp + ref, padding=True, truncation=True, return_tensors="pt"
                )
                model_output = evaluator(**encoded_input)

                # Perform pooling
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)
                # print("sentence_embeddings_norm=", sentence_embeddings_norm)
                text_scores = []
                for i in range(len(hyp)):
                    text_score = (
                        cosine_similarity(
                            sentence_embeddings_norm[i : i + 1],
                            sentence_embeddings_norm[len(hyp) + i : len(hyp) + i + 1],
                        )
                        .cpu()
                        .detach()
                        .numpy()
                        .max()
                    )
                    text_scores.append(text_score)
            return text_scores

    else:
        assert False

    return compute_similarity, evaluator
