#!/usr/bin/env python3
"""
Run evaluation: for each item in data/eval/questions.json, retrieve -> answer -> compare to reference.

Metrics: exact_match, ref_in_pred; optionally --llm-judge. Reports accuracy.

Run from project root (LLM in .env):
    python scripts/run_eval.py
    python scripts/run_eval.py --llm-judge
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress urllib3/OpenSSL warning on macOS (LibreSSL vs OpenSSL)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT))

from openai import AuthenticationError, NotFoundError, RateLimitError

from src.config.settings import (
    CHUNKS_JSONL,
    INDEX_PATH,
    PROJECT_ROOT,
    RERANK_RETRIEVE_K,
    TOP_K,
    USE_RERANKER,
)
from src.retrieval.reranker import rerank
from src.evaluation import exact_match, embedding_similarity, bleu, llm_judge, ref_in_pred, rouge_l_f1
from src.generation import answer
from src.retrieval import Retriever, embed_query, embed_texts
from src.retrieval.vector_store import VectorStore

QUESTIONS_JSON = PROJECT_ROOT / "data" / "eval" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"


def build_retriever():
    if INDEX_PATH.exists():
        return Retriever(INDEX_PATH)
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"Need {CHUNKS_JSONL} or {INDEX_PATH}. Run: python scripts/build_chunks.py")
    chunks_list = []
    with open(CHUNKS_JSONL) as f:
        for line in f:
            if line.strip():
                chunks_list.append(json.loads(line))
    texts = [c["text"] for c in chunks_list]
    embs = embed_texts(texts)
    store = VectorStore()
    store.add(embs, chunks_list)

    class _InMemoryRetriever:
        def retrieve(self, q, top_k=5):
            return store.search(embed_query(q), top_k=top_k)

    return _InMemoryRetriever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAG evaluation on data/eval/questions.json")
    ap.add_argument("--llm-judge", action="store_true", help="Also run LLM-as-judge (extra API calls)")
    ap.add_argument("--no-save", action="store_true", help="Do not persist results to data/eval/results/")
    args = ap.parse_args()

    if not QUESTIONS_JSON.exists():
        print(f"Eval questions not found: {QUESTIONS_JSON}")
        sys.exit(1)

    with open(QUESTIONS_JSON) as f:
        questions = json.load(f)

    try:
        retriever = build_retriever()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    start = datetime.now(timezone.utc)
    results = []
    for item in questions:
        qid, q, ref = item["id"], item["question"], item["answer"]
        retrieve_k = RERANK_RETRIEVE_K if USE_RERANKER else TOP_K
        chunks = retriever.retrieve(q, top_k=retrieve_k)
        if USE_RERANKER and len(chunks) > TOP_K:
            chunks = rerank(q, chunks, top_k=TOP_K)
        try:
            pred = answer(q, context_chunks=chunks)
        except (ValueError, AuthenticationError, NotFoundError, RateLimitError) as e:
            pred = ""
            print(f"[{qid}] Error: {e}")
        em = exact_match(pred, ref)
        rp = ref_in_pred(pred, ref)
        try:
            sim = embedding_similarity(pred, ref)
        except Exception as e:
            sim = -1.0
            print(f"[{qid}] embedding_similarity error: {e}")
        try:
            bleu_val = bleu(pred, ref)
        except Exception as e:
            bleu_val = -1.0
            print(f"[{qid}] bleu error: {e}")
        try:
            rouge_val = rouge_l_f1(pred, ref)
        except Exception as e:
            rouge_val = -1.0
            print(f"[{qid}] rouge error: {e}")
        row = {
            "id": qid, "question": q, "ref": ref, "pred": pred,
            "exact_match": em, "ref_in_pred": rp,
            "embedding_similarity": round(sim, 4),
            "bleu": round(bleu_val, 4) if bleu_val >= 0 else None,
            "rouge_l_f1": round(rouge_val, 4) if rouge_val >= 0 else None,
        }
        if args.llm_judge:
            try:
                row["llm_judge"] = llm_judge(q, ref, pred)
            except Exception as e:
                row["llm_judge"] = False
                print(f"[{qid}] llm_judge error: {e}")
        results.append(row)

    n = len(results)
    em_correct = sum(1 for r in results if r["exact_match"])
    rp_correct = sum(1 for r in results if r["ref_in_pred"])

    print("Per-question:")
    print("id  exact  ref_in_pred" + ("  llm_judge" if args.llm_judge else "") + "  question (truncated)")
    for r in results:
        extra = f"  {r['llm_judge']!s:>9}" if args.llm_judge else ""
        print(f"{r['id']:>2}  {r['exact_match']!s:>5}  {r['ref_in_pred']!s:>11}{extra}  {r['question'][:50]}...")

    mean_sim = sum(r.get("embedding_similarity", -1) for r in results) / n
    mean_sim = mean_sim if mean_sim >= 0 else float("nan")
    bleu_vals = [r["bleu"] for r in results if r.get("bleu") is not None]
    rouge_vals = [r["rouge_l_f1"] for r in results if r.get("rouge_l_f1") is not None]
    mean_bleu = sum(bleu_vals) / len(bleu_vals) if bleu_vals else float("nan")
    mean_rouge = sum(rouge_vals) / len(rouge_vals) if rouge_vals else float("nan")
    print()
    print(f"Overall: exact_match={em_correct}/{n} ({100 * em_correct / n:.1f}%); ref_in_pred={rp_correct}/{n} ({100 * rp_correct / n:.1f}%)")
    print(f"         mean_embedding_similarity={mean_sim:.4f}; mean_bleu={mean_bleu:.4f}; mean_rouge_l_f1={mean_rouge:.4f}")
    if args.llm_judge:
        lj_correct = sum(1 for r in results if r.get("llm_judge"))
        print(f"         llm_judge={lj_correct}/{n} ({100 * lj_correct / n:.1f}%)")

    # Persist results
    if not args.no_save:
        end = datetime.now(timezone.utc)
        payload = {
            "timestamp_utc": start.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            "n": n,
            "exact_match": {"correct": em_correct, "pct": round(100 * em_correct / n, 1)},
            "ref_in_pred": {"correct": rp_correct, "pct": round(100 * rp_correct / n, 1)},
            "mean_embedding_similarity": round(mean_sim, 4) if not (mean_sim != mean_sim) else None,
            "mean_bleu": round(mean_bleu, 4) if bleu_vals else None,
            "mean_rouge_l_f1": round(mean_rouge, 4) if rouge_vals else None,
            "results": results,
        }
        if args.llm_judge:
            lj_correct = sum(1 for r in results if r.get("llm_judge"))
            payload["llm_judge"] = {"correct": lj_correct, "pct": round(100 * lj_correct / n, 1)}

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = start.strftime("%Y-%m-%d_%H-%M-%S")
        path_ts = RESULTS_DIR / f"{ts}.json"
        path_latest = RESULTS_DIR / "latest.json"
        with open(path_ts, "w") as f:
            json.dump(payload, f, indent=2)
        with open(path_latest, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {path_ts} and {path_latest}")


if __name__ == "__main__":
    main()
