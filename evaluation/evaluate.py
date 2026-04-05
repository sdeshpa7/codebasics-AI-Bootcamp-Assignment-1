"""
evaluation/evaluate.py
======================
RAGAS-based evaluation of the FinSolve RAG pipeline with an ablation study.

Architecture components under test
───────────────────────────────────
A  Full pipeline          input-guards + semantic-routing + RBAC + retrieval + LLM + output-guards
B  No input guardrails    semantic-routing + RBAC + retrieval + LLM + output-guards
C  No semantic routing    RBAC + retrieval (no query-expansion) + LLM + output-guards
D  No RBAC filter         full routing + retrieval (no Qdrant role filter) + LLM + output-guards
E  No output guardrails   input-guards + routing + RBAC + retrieval + LLM (raw)
F  Naive RAG              retrieval only (no guardrails, no routing) + LLM

RAGAS metrics (v0.4.x)   — each returns a value in [0, 1]
───────────────────────────────────────────────────────────
- Faithfulness            Is the answer grounded in the retrieved context?
- Answer Relevancy        Is the answer relevant to the question?
- Context Precision       Were retrieved chunks actually useful?
- Context Recall          Did retrieval capture all relevant information?
- Overall Score           Arithmetic mean of the four metrics above

The evaluator LLM is llama3-8b-8192 via Groq — a different model from
the pipeline's generation model (openai/gpt-oss-20b) to avoid self-grading bias.

Usage
─────
# Full evaluation (all ablation variants):
    .venv/bin/python evaluation/evaluate.py

# Quick test (first N questions only):
    .venv/bin/python evaluation/evaluate.py --limit 5

# Single variant:
    .venv/bin/python evaluation/evaluate.py --variant A

# Resume a previously interrupted run:
    .venv/bin/python evaluation/evaluate.py --resume
"""

import argparse
import asyncio
import json
import sys
import os
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)

from app.config import settings
from app.embeddings import _get_model as get_embed_model
from app.vector_store import get_client as get_qdrant_client
from app.semantic_routing import build_router, get_qdrant_filter
from app.guardrails import InputGuardrails, apply_output_guardrails
from app.llm import answer_query
from app.chunking import Chunk

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR    = Path(__file__).resolve().parent
GT_FILE     = EVAL_DIR / "ground_truth_qa.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Evaluator LLM (separate from pipeline to avoid self-grading bias) ─────────────
# llama3-8b-8192 decommissioned — use llama-3.1-8b-instant (fast, lightweight)
EVALUATOR_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# Metric names in result dicts / CSV columns
METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator setup
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics():
    """Instantiate the 4 RAGAS metric objects. Uses AsyncOpenAI for Groq compatibility."""
    # AsyncOpenAI is required: RAGAS 0.4.x collections metrics call agenerate() internally
    groq_async_client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    eval_llm = llm_factory(EVALUATOR_MODEL, provider="openai", client=groq_async_client)
    eval_emb = embedding_factory(provider="huggingface", model=EMBED_MODEL)

    return {
        "faithfulness":      Faithfulness(llm=eval_llm),
        "answer_relevancy":  AnswerRelevancy(llm=eval_llm, embeddings=eval_emb),
        "context_precision": ContextPrecisionWithReference(llm=eval_llm),
        "context_recall":    ContextRecall(llm=eval_llm),
    }


async def _score_one(metrics: dict, question: str, answer: str,
                     contexts: list[str], ground_truth: str,
                     inter_metric_delay: float = 5.0) -> dict[str, float]:
    """
    Score one QA triple using each metric's ascore() sequentially to avoid
    hitting the Groq free-tier 6K TPM rate limit.
    Each value is in [0, 1].
    """
    calls = [
        ("faithfulness",      lambda: metrics["faithfulness"].ascore(
                                user_input=question, response=answer, retrieved_contexts=contexts)),
        ("answer_relevancy",  lambda: metrics["answer_relevancy"].ascore(
                                user_input=question, response=answer)),
        ("context_precision", lambda: metrics["context_precision"].ascore(
                                user_input=question, reference=ground_truth, retrieved_contexts=contexts)),
        ("context_recall",    lambda: metrics["context_recall"].ascore(
                                user_input=question, retrieved_contexts=contexts, reference=ground_truth)),
    ]
    scores = {}
    for i, (key, make_coro) in enumerate(calls):
        try:
            result = await make_coro()
            scores[key] = round(float(result.value), 4)
        except Exception as e:
            print(f"\n      ⚠ {key}: {str(e)[:120]}")
            scores[key] = None
        # Brief pause between metric calls to respect TPM limits
        if i < len(calls) - 1:
            await asyncio.sleep(inter_metric_delay)
    return scores


def score_sample(metrics: dict, question: str, answer: str,
                 contexts: list[str], ground_truth: str) -> dict[str, float]:
    """
    Synchronous entry-point. Runs the async _score_one() via asyncio.run().
    Each returned value is in [0, 1] (or None on transient LLM error).
    """
    return asyncio.run(_score_one(metrics, question, answer, contexts, ground_truth))


# ─────────────────────────────────────────────────────────────────────────────
# Role helpers
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_ROLES = {
    "engineering": ["C-Level", "General", "Engineering"],
    "finance":     ["C-Level", "General", "Finance"],
    "hr":          ["C-Level", "General", "HR"],
    "marketing":   ["C-Level", "General", "Marketing"],
    "general":     ["C-Level", "General"],
}

ROUTE_PREFIX = {
    "HR":          "HR policy employee handbook leave vacation benefits: ",
    "Finance":     "finance report budget revenue expense forecast: ",
    "Engineering": "engineering technical SLA incident sprint architecture: ",
    "Marketing":   "marketing campaign lead ROI acquisition brand: ",
    "C-Level":     "executive summary strategy performance overview: ",
}


def _role_for(collection: str) -> list[str]:
    return COLLECTION_ROLES.get(collection.lower(), ["C-Level", "General"])


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(query: str, user_roles: list[str], top_k: int = 8,
              apply_rbac: bool = True, route: Optional[str] = None,
              use_query_expansion: bool = True) -> tuple[list[Chunk], list[str]]:

    embed_model = get_embed_model()
    expanded    = f"{ROUTE_PREFIX.get(route or '', '')}{query}" if use_query_expansion else query
    q_vec       = embed_model.encode([expanded], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    q_filter    = get_qdrant_filter(user_roles) if apply_rbac else None
    qdrant      = get_qdrant_client()

    result = qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=q_vec,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
        score_threshold=0.1,
    )

    chunks, texts = [], []
    for hit in result.points:
        p = hit.payload
        chunks.append(Chunk(
            id=str(hit.id), text=p.get("text", ""),
            chunk_type=p.get("chunk_type", "text"),
            page_number=p.get("page_number"),
            section_title=p.get("section_title"),
            parent_chunk_id=p.get("parent_chunk_id"),
            source_file=p.get("source_file", "unknown"),
            collection=p.get("collection", "general"),
            access_roles=p.get("access_roles", []),
        ))
        texts.append(p.get("text", ""))

    return chunks, texts


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline variants
# ─────────────────────────────────────────────────────────────────────────────

_router  = None
_guards  = None


def _get_router():
    global _router
    if _router is None:
        _router = build_router()
    return _router


def _get_guards():
    global _guards
    if _guards is None:
        _guards = InputGuardrails(session_limit=99999)
    return _guards


def _run_item(q: dict, apply_input_guards=True, apply_routing=True,
              apply_rbac=True, apply_output_guards=True) -> Optional[dict]:
    """Generic pipeline runner — toggle components via boolean flags."""
    question   = q["question"]
    session_id = str(uuid.uuid4())
    user_roles = _role_for(q["collection"])

    route_name = None

    if apply_routing:
        router      = _get_router()
        route_match = router.route(question, user_roles)
        route_name  = route_match.route

        if apply_input_guards:
            guard_result = _get_guards().check(
                question, session_id=session_id,
                user_access_roles=user_roles, route_score=route_match.score
            )
            if not guard_result.passed:
                return None

        if not route_match.allowed:
            return None
    else:
        if apply_input_guards:
            guard_result = _get_guards().check(
                question, session_id=session_id,
                user_access_roles=user_roles, route_score=0.5
            )
            if not guard_result.passed:
                return None

    chunks, contexts = _retrieve(
        question, user_roles,
        apply_rbac=apply_rbac,
        route=route_name,
        use_query_expansion=apply_routing,
    )
    if not chunks:
        return None

    raw_answer = answer_query(question, chunks, user_roles)

    if apply_output_guards:
        output = apply_output_guardrails(raw_answer, chunks, user_roles)
        answer = output.response
    else:
        answer = raw_answer

    return {"question": question, "answer": answer,
            "contexts": contexts, "ground_truth": q["ground_truth"]}


VARIANTS = {
    "A": ("Full Pipeline (baseline)",
          lambda q: _run_item(q, True, True, True, True)),
    "B": ("No Input Guardrails",
          lambda q: _run_item(q, False, True, True, True)),
    "C": ("No Semantic Routing / Query Expansion",
          lambda q: _run_item(q, True, False, True, True)),
    "D": ("No RBAC Filter",
          lambda q: _run_item(q, True, True, False, True)),
    "E": ("No Output Guardrails",
          lambda q: _run_item(q, True, True, True, False)),
    "F": ("Naive RAG (no guardrails, no RBAC)",
          lambda q: _run_item(q, False, False, False, False)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _overall(row: dict) -> Optional[float]:
    vals = [row.get(k) for k in METRIC_KEYS if row.get(k) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(label: str, description: str, fn,
                questions: list[dict], metrics: dict,
                delay: float = 1.5) -> dict:
    """
    Run one ablation variant:
    1. Call the pipeline fn() for every question.
    2. Score each answer with RAGAS metrics.
    3. Average across the dataset → store mean + per-sample rows.
    """
    print(f"\n{'═'*68}")
    print(f"  Variant {label}: {description}")
    print(f"  Questions: {len(questions)}")
    print(f"{'═'*68}")

    per_sample = []
    skipped    = 0

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"Q{i:03d}")
        print(f"  [{i:02d}/{len(questions)}] [{qid}] {q['question'][:55]}…", end=" ", flush=True)

        try:
            rec = fn(q)
            if rec is None:
                print("SKIP")
                skipped += 1
                continue

            scores = score_sample(
                metrics,
                question    = rec["question"],
                answer      = rec["answer"],
                contexts    = rec["contexts"],
                ground_truth= rec["ground_truth"],
            )
            scores["overall"] = _overall(scores)
            scores["id"]      = qid
            scores["collection"] = q.get("collection", "?")

            per_sample.append(scores)
            line = "  ".join(
                f"{k[:6]}={v:.3f}" if v is not None else f"{k[:6]}=N/A"
                for k, v in scores.items() if k in METRIC_KEYS
            )
            print(f"OK  [{line}]")

        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1

        time.sleep(delay)

    # ── Aggregate ────────────────────────────────────────────────────────────
    if not per_sample:
        print(f"\n  ⚠ No valid samples — all {skipped} were skipped.")
        return _empty_result(label, description, skipped)

    agg = {k: _mean([s.get(k) for s in per_sample]) for k in METRIC_KEYS}
    agg["overall"] = _overall(agg)

    result = {
        "variant":          label,
        "description":      description,
        "n_evaluated":      len(per_sample),
        "n_skipped":        skipped,
        "faithfulness":     agg["faithfulness"],
        "answer_relevancy": agg["answer_relevancy"],
        "context_precision":agg["context_precision"],
        "context_recall":   agg["context_recall"],
        "overall_score":    agg["overall"],
        "per_sample":       per_sample,
    }

    print(f"\n  ✅ Variant {label} — averaged over {len(per_sample)} samples:")
    f = lambda v: f"{v:.4f}" if v is not None else " N/A  "
    print(f"     Faithfulness      : {f(agg['faithfulness'])}")
    print(f"     Answer Relevancy  : {f(agg['answer_relevancy'])}")
    print(f"     Context Precision : {f(agg['context_precision'])}")
    print(f"     Context Recall    : {f(agg['context_recall'])}")
    print(f"     ─────────────────────────────────")
    print(f"     Overall Score     : {f(agg['overall'])}  ← mean of above")

    return result


def _empty_result(label, desc, skipped):
    return {
        "variant": label, "description": desc,
        "n_evaluated": 0, "n_skipped": skipped,
        "faithfulness": None, "answer_relevancy": None,
        "context_precision": None, "context_recall": None,
        "overall_score": None, "per_sample": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print ablation table
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(results: list[dict]):
    """Print a comparison table with Δ deltas relative to Variant A."""
    full = next((r for r in results if r["variant"] == "A"), None)

    print(f"\n\n{'═'*84}")
    print("  ABLATION STUDY — FINAL RESULTS")
    print(f"  Each metric is in [0, 1]. Overall = arithmetic mean of all four.")
    print(f"{'═'*84}")
    header = (
        f"  {'Var':<4} {'Description':<38} "
        f"{'Faith':>6} {'AnsRel':>6} {'CtxPre':>6} {'CtxRec':>6} {'Overall':>7}  {'N':>4}"
    )
    print(header)
    print(f"  {'-'*80}")

    def fmt(v): return f"{v:.4f}" if v is not None else " N/A  "
    def delta(new, base):
        if new is None or base is None: return "  N/A  "
        d = new - base
        return ("+" if d >= 0 else "") + f"{d:.4f}"

    for r in results:
        print(
            f"  {r['variant']:<4} {r['description']:<38} "
            f"{fmt(r['faithfulness']):>6} {fmt(r['answer_relevancy']):>6} "
            f"{fmt(r['context_precision']):>6} {fmt(r['context_recall']):>6} "
            f"{fmt(r['overall_score']):>7}  {r['n_evaluated']:>4}"
        )
        if full and r["variant"] != "A":
            dline = (
                f"  {'Δ vs A':.<42} "
                f"{delta(r['faithfulness'],   full['faithfulness']):>6} "
                f"{delta(r['answer_relevancy'],full['answer_relevancy']):>6} "
                f"{delta(r['context_precision'],full['context_precision']):>6} "
                f"{delta(r['context_recall'],  full['context_recall']):>6} "
                f"{delta(r['overall_score'],   full['overall_score']):>7}"
            )
            print(dline)

    print(f"{'═'*84}")

    # Component contribution table
    if full and full["overall_score"] is not None:
        print(f"\n  COMPONENT CONTRIBUTION  (how much each component adds to Overall Score)")
        print(f"  {'Component':<40} {'Δ Overall':>10}")
        print(f"  {'-'*52}")
        labels = {
            "B": "Input Guardrails",
            "C": "Semantic Routing / Query Expansion",
            "D": "RBAC Filtering",
            "E": "Output Guardrails",
        }
        for r in results:
            if r["variant"] in labels and r["overall_score"] is not None:
                contrib = full["overall_score"] - r["overall_score"]
                print(f"  {labels[r['variant']]:<40} {'+' if contrib >= 0 else ''}{contrib:.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinSolve RAG RAGAS Ablation Evaluation")
    parser.add_argument("--limit",   type=int,   default=None, help="Evaluate first N questions only")
    parser.add_argument("--variant", type=str,   default=None, help="Run a single variant: A-F")
    parser.add_argument("--delay",   type=float, default=5.0,  help="Seconds between samples (default 5.0)")
    parser.add_argument("--resume",  action="store_true",      help="Skip variants with existing results")
    args = parser.parse_args()

    # ── Load ground truth ────────────────────────────────────────────────────
    print(f"\nLoading ground truth from: {GT_FILE}")
    with open(GT_FILE) as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[: args.limit]
        print(f"⚡ Limited to {args.limit} questions.")

    col_counts = {}
    for q in questions:
        col_counts[q.get("collection", "?")] = col_counts.get(q.get("collection", "?"), 0) + 1
    print(f"\nTotal questions: {len(questions)}")
    for col, n in sorted(col_counts.items()):
        print(f"  {col:<15} {n} questions")

    # ── Initialise RAGAS metrics (once, shared across all variants) ──────────
    print(f"\nInitialising RAGAS metrics…")
    print(f"  Evaluator LLM : {EVALUATOR_MODEL}  (Groq — different from pipeline model)")
    print(f"  Pipeline LLM  : {settings.groq_model}")
    metrics = build_metrics()
    print("  Metrics ready :", list(metrics.keys()))

    # ── Select variants ──────────────────────────────────────────────────────
    variants_to_run = (
        {args.variant: VARIANTS[args.variant]}
        if args.variant
        else VARIANTS
    )

    # ── Run ablation ─────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for label, (description, fn) in variants_to_run.items():
        result_file = RESULTS_DIR / f"variant_{label}_{ts}.json"

        if args.resume:
            existing = sorted(RESULTS_DIR.glob(f"variant_{label}_*.json"))
            if existing:
                print(f"\n  ↩ Resuming Variant {label} from {existing[-1].name}")
                with open(existing[-1]) as f:
                    all_results.append(json.load(f))
                continue

        result = run_variant(label, description, fn, questions, metrics, delay=args.delay)
        all_results.append(result)

        # Save per-variant (strip per_sample to keep it manageable)
        slim = {k: v for k, v in result.items() if k != "per_sample"}
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved  → {result_file.name}")

    # ── Print summary table ──────────────────────────────────────────────────
    print_ablation_table(all_results)

    # ── Save combined summary ────────────────────────────────────────────────
    summary_json = RESULTS_DIR / f"ablation_summary_{ts}.json"
    summary_csv  = RESULTS_DIR / f"ablation_summary_{ts}.csv"

    summary_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "per_sample"}
        summary_rows.append(row)

    with open(summary_json, "w") as f:
        json.dump({
            "timestamp":       ts,
            "n_questions":     len(questions),
            "evaluator_model": EVALUATOR_MODEL,
            "pipeline_model":  settings.groq_model,
            "metric_range":    "0.0 – 1.0 (higher is better)",
            "results":         summary_rows,
        }, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"\n  📁 Results saved:")
    print(f"     JSON → {summary_json}")
    print(f"     CSV  → {summary_csv}")
    print(f"\n  ✅ Evaluation complete.\n")


if __name__ == "__main__":
    main()
