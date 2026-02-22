"""Evaluation metrics: exact match, ref-in-pred, LLM-as-judge, embedding similarity."""

import re
from src.generation import generate


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, ref: str) -> bool:
    """True if normalized prediction equals normalized reference."""
    return _normalize(pred) == _normalize(ref)


def ref_in_pred(pred: str, ref: str) -> bool:
    """True if the reference appears as a substring in the prediction (case-insensitive)."""
    np, nr = _normalize(pred), _normalize(ref)
    if not nr:
        return True
    return nr in np


def llm_judge(question: str, ref: str, pred: str) -> bool:
    """
    Use the LLM to judge whether the prediction contains the key information from the reference.
    Returns True if the LLM responds with YES (or similar). Uses generate() from src.generation.
    """
    prompt = f"""You are a judge. Given:
Question: {question}
Reference answer: {ref}
Model prediction: {pred}

Does the prediction contain the key information from the reference? Be lenient: if the main facts are present, say YES. Reply with only YES or NO."""
    out = generate(prompt).strip().upper()
    return out.startswith("YES")


def embedding_similarity(pred: str, ref: str) -> float:
    """
    Cosine similarity between embeddings of pred and ref. Returns value in [-1, 1] (higher = more similar).
    Uses the same embedding model as retrieval (sentence-transformers).
    """
    if not (pred or ref):
        return 1.0
    if not pred or not ref:
        return 0.0
    from src.retrieval import embed_query
    import numpy as np
    ep = embed_query(pred)
    er = embed_query(ref)
    return float(np.dot(ep, er))  # already L2-normalized, so dot = cosine sim


def _ensure_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def bleu(pred: str, ref: str) -> float:
    """
    BLEU-4 score (0–1) between prediction and reference. Uses nltk.
    """
    if not ref:
        return 1.0 if not pred else 0.0
    _ensure_nltk()
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tok = nltk.word_tokenize(ref.lower())
    pred_tok = nltk.word_tokenize((pred or "").lower())
    if not ref_tok:
        return 1.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth))


def rouge_l_f1(pred: str, ref: str) -> float:
    """
    ROUGE-L F1 (0–1) between prediction and reference. Uses rouge_score.
    """
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(ref, pred)
    return scores["rougeL"].fmeasure
