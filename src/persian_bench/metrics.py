import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def normalize_persian(text: str) -> str:
    """
    Standard Persian text normalization:
    - Normalizes Arabic Yeh/Kaf to standard Persian characters
    - Converts all Persian/Arabic numerals to ASCII standard
    - Removes decorative tatweel, diacritics, and redundant whitespaces
    """
    if not text:
        return ""

    # Digits
    for i, (p_d, a_d) in enumerate(zip(PERSIAN_DIGITS, ARABIC_DIGITS)):
        text = text.replace(p_d, str(i)).replace(a_d, str(i))

    # Characters
    text = (
        text.replace("ك", "ک")
        .replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ة", "ه")
        .replace("ـ", "")
        .replace("\u200c", " ")  # Normalize ZWNJ to space for token matching
    )

    # Remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Strip punctuation and collapse whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM) metric after canonical Persian normalization."""
    return 1.0 if normalize_persian(prediction) == normalize_persian(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Computes token-level Precision, Recall, and F1 score over normalized Persian tokens.
    """
    pred_tokens = normalize_persian(prediction).split()
    truth_tokens = normalize_persian(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        score = 1.0 if pred_tokens == truth_tokens else 0.0
        return score, score, score

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def check_hallucination_contamination(prediction: str, distractors: List[str]) -> bool:
    """
    Evaluates whether the generation regurgitates known deceptive falsehoods.
    Returns True if hallucination was detected, False if truthful.
    """
    norm_pred = normalize_persian(prediction)
    for distractor in distractors:
        norm_dist = normalize_persian(distractor)
        if norm_dist and norm_dist in norm_pred:
            return True
    return False


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    distractors_list: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Evaluates a batch of generations against ground-truth references.
    Calculates: Exact Match, Macro Precision, Macro Recall, Macro F1, Truthfulness Rate.
    """
    assert len(predictions) == len(references), "Predictions and references length mismatch"

    em_scores = []
    precisions = []
    recalls = []
    f1s = []
    hallucination_detected = 0
    total_with_distractors = 0

    for i, (pred, truth) in enumerate(zip(predictions, references)):
        em_scores.append(compute_exact_match(pred, truth))
        p, r, f = compute_token_f1(pred, truth)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        if distractors_list and distractors_list[i]:
            total_with_distractors += 1
            if check_hallucination_contamination(pred, distractors_list[i]):
                hallucination_detected += 1

    n = max(1, len(predictions))
    truthfulness_rate = (
        1.0 - (hallucination_detected / total_with_distractors)
        if total_with_distractors > 0
        else 1.0
    )

    return {
        "exact_match": round(sum(em_scores) / n, 4),
        "macro_precision": round(sum(precisions) / n, 4),
        "macro_recall": round(sum(recalls) / n, 4),
        "macro_f1": round(sum(f1s) / n, 4),
        "truthfulness_rate": round(truthfulness_rate, 4),
        "samples_evaluated": len(predictions),
    }
