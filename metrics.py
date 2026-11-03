"""
Standard Evaluation Metrics for Video & Image Captioning
Computes BLEU-1..4, METEOR, ROUGE-L, CIDEr, and Action Recognition Accuracy.
"""

from collections import Counter, defaultdict
import math
import re
from typing import Dict, List, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def tokenize(text: str) -> List[str]:
    """Basic lowercase alphanumeric tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts n-grams of order n as a Counter."""
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


class CaptionMetrics:
    """Computes academic benchmark metrics across predicted vs reference captions."""

    def __init__(self):
        self.smoother = SmoothingFunction().method1
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute_sentence_metrics(
        self, prediction: str, references: List[str]
    ) -> Dict[str, float]:
        """
        Computes BLEU (1-4), ROUGE-L, and METEOR for a single prediction
        against a list of ground-truth reference captions.
        """
        pred_tokens = tokenize(prediction)
        ref_tokens_list = [tokenize(ref) for ref in references if ref.strip()]

        if not pred_tokens or not ref_tokens_list:
            return {
                "BLEU-1": 0.0,
                "BLEU-2": 0.0,
                "BLEU-3": 0.0,
                "BLEU-4": 0.0,
                "ROUGE-L": 0.0,
                "METEOR": 0.0,
            }

        # BLEU 1 to 4
        weights_dict = {
            "BLEU-1": (1.0, 0.0, 0.0, 0.0),
            "BLEU-2": (0.5, 0.5, 0.0, 0.0),
            "BLEU-3": (0.333, 0.333, 0.333, 0.0),
            "BLEU-4": (0.25, 0.25, 0.25, 0.25),
        }
        bleu_scores = {}
        for metric_name, weights in weights_dict.items():
            score = sentence_bleu(
                ref_tokens_list,
                pred_tokens,
                weights=weights,
                smoothing_function=self.smoother,
            )
            bleu_scores[metric_name] = score * 100.0  # Percentage scale standard in literature

        # METEOR
        try:
            meteor = meteor_score(ref_tokens_list, pred_tokens) * 100.0
        except Exception:
            meteor = 0.0

        # ROUGE-L: best score against any reference
        best_rouge_l = 0.0
        for ref in references:
            score = self.rouge_scorer.score(ref, prediction)["rougeL"].fmeasure * 100.0
            if score > best_rouge_l:
                best_rouge_l = score

        return {
            **bleu_scores,
            "ROUGE-L": best_rouge_l,
            "METEOR": meteor,
        }

    @staticmethod
    def compute_corpus_cider(
        predictions: Dict[str, str], references: Dict[str, List[str]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes CIDEr (Consensus-based Image Description Evaluation)
        across the entire dataset corpus.
        Returns: (overall_cider, per_item_cider_dict)
        """
        # Document frequencies across references
        num_items = len(references)
        if num_items == 0:
            return 0.0, {}

        df_counts = [defaultdict(int) for _ in range(4)]
        ref_tokens_map = {}
        pred_tokens_map = {}

        for item_id, ref_list in references.items():
            tokenized_refs = [tokenize(r) for r in ref_list if r.strip()]
            ref_tokens_map[item_id] = tokenized_refs
            pred_tokens_map[item_id] = tokenize(predictions.get(item_id, ""))

            # Count document frequency for each n-gram
            seen_ngrams = [set() for _ in range(4)]
            for ref in tokenized_refs:
                for n in range(4):
                    ngrams = get_ngrams(ref, n + 1)
                    seen_ngrams[n].update(ngrams.keys())

            for n in range(4):
                for ngram in seen_ngrams[n]:
                    df_counts[n][ngram] += 1

        # TF-IDF vectors calculation
        def get_tfidf_vector(tokens: List[str], n: int) -> Dict[Tuple, float]:
            counts = get_ngrams(tokens, n + 1)
            total = sum(counts.values())
            vec = {}
            if total == 0:
                return vec
            for ngram, count in counts.items():
                df = df_counts[n].get(ngram, 1)
                # Smoothed IDF
                idf = math.log(max(1.0, (num_items + 1.0) / (df + 1.0)))
                vec[ngram] = (count / total) * idf
            return vec

        def cosine_similarity(v1: Dict[Tuple, float], v2: Dict[Tuple, float]) -> float:
            dot = sum(v1[k] * v2[k] for k in v1 if k in v2)
            norm1 = math.sqrt(sum(v * v for v in v1.values()))
            norm2 = math.sqrt(sum(v * v for v in v2.values()))
            if norm1 * norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        per_item_cider = {}
        for item_id, tokenized_refs in ref_tokens_map.items():
            pred_tokens = pred_tokens_map.get(item_id, [])
            if not pred_tokens or not tokenized_refs:
                per_item_cider[item_id] = 0.0
                continue

            n_scores = []
            for n in range(4):
                pred_vec = get_tfidf_vector(pred_tokens, n)
                ref_sims = []
                for ref in tokenized_refs:
                    ref_vec = get_tfidf_vector(ref, n)
                    ref_sims.append(cosine_similarity(pred_vec, ref_vec))
                avg_sim = sum(ref_sims) / len(ref_sims) if ref_sims else 0.0
                n_scores.append(avg_sim)

            # Standard CIDEr scale: average across 1-4 grams * 10
            cider_score = (sum(n_scores) / 4.0) * 10.0
            per_item_cider[item_id] = cider_score

        eval_keys = [k for k, pred in predictions.items() if pred and pred.strip() and k in per_item_cider]
        overall_cider = (
            sum(per_item_cider[k] for k in eval_keys) / len(eval_keys) if eval_keys else 0.0
        )
        return overall_cider, per_item_cider
