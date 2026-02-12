"""
rag_engine.py  ·  TruPharma RAG Back-End
=========================================
Wraps openfda_rag helpers with:
  - Hybrid retrieval (dense + sparse + optional rerank)
  - LLM-grounded answer generation (Google Gemini or extractive fallback)
  - Interaction logging to logs/product_metrics.csv
"""

import gc
import os
import sys
import re
import csv
import time
import json
import warnings
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Make sibling modules importable ──────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openfda_rag import (
    TextChunk,
    SubChunk,
    build_artifacts,
    build_openfda_query,
    tokenize,
)

# ── Silence noisy libraries ──────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

FIELD_ALLOWLIST: List[str] = [
    "active_ingredient",
    "description",
    "dosage_and_administration",
    "drug_interactions",
    "information_for_patients",
    "when_using",
    "overdosage",
    "stop_use",
    "user_safety_warnings",
    "warnings",
]
FIELD_BLOCKLIST = {"spl_product_data_elements"}

API_BASE = "https://api.fda.gov/drug/label.json"
DEFAULT_LIMIT = 20           # records per API page (keep small for cloud memory)
DEFAULT_MAX_REC = 20         # total records to pull (enough for evidence, fits in 1GB)
USE_SENTENCE_TRANSFORMERS = False   # False = TF-IDF (fast); True = dense (slow on CPU)

# ── Logging paths ────────────────────────────────────────────
_PROJECT_ROOT = _SRC.parent
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_CSV = LOG_DIR / "product_metrics.csv"
LOG_COLS = [
    "timestamp",
    "query",
    "latency_ms",
    "evidence_ids",
    "confidence",
    "num_evidence",
    "num_records",
    "retrieval_method",
    "llm_used",
    "answer_preview",
]


# ══════════════════════════════════════════════════════════════
#  RETRIEVAL HELPERS
# ══════════════════════════════════════════════════════════════

def _embed_query(query, embedder_type, embedder_model, vectorizer):
    """Embed a query using the same method that was used during indexing."""
    if embedder_type == "sentence_transformers":
        try:
            from openfda_rag import _get_st_model
        except ImportError:
            return None
        name = embedder_model or "sentence-transformers/all-MiniLM-L6-v2"
        return _get_st_model(name).encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
    if embedder_type == "tfidf" and vectorizer is not None:
        from sklearn.preprocessing import normalize as sk_normalize
        return sk_normalize(vectorizer.transform([query])).toarray().astype(np.float32)
    return None


def _dense(query, index, corpus, e_type, e_model, vec, k=15):
    """Dense (vector) search via FAISS inner-product index."""
    if index is None or not corpus:
        return []
    qv = _embed_query(query, e_type, e_model, vec)
    if qv is None:
        return []
    n = min(k, index.ntotal)
    scores, idxs = index.search(qv.astype(np.float32), n)
    return [
        (float(s), corpus[int(i)])
        for s, i in zip(scores[0], idxs[0])
        if int(i) >= 0
    ]


def _sparse(query, bm25, corpus, k=15):
    """Sparse (BM25) keyword search."""
    if bm25 is None or not corpus:
        return []
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), corpus[int(i)]) for i in top]


def _fuse(dense_res, sparse_res, alpha=0.5, k=15):
    """Reciprocal-rank fusion of dense + sparse results."""
    cid = lambda it: getattr(it, "chunk_id", str(it))
    dr = {cid(it): r for r, (_, it) in enumerate(dense_res, 1)}
    sr = {cid(it): r for r, (_, it) in enumerate(sparse_res, 1)}
    bucket: Dict[str, Any] = {}
    for _, it in list(dense_res) + list(sparse_res):
        bucket.setdefault(cid(it), it)
    fused = []
    for key, obj in bucket.items():
        d = dr.get(key, len(dense_res) + 1)
        s = sr.get(key, len(sparse_res) + 1)
        fused.append((alpha / d + (1 - alpha) / s, obj))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused[:k]


def _try_rerank(query, items, top_k):
    """Best-effort cross-encoder rerank; falls back silently."""
    try:
        from sentence_transformers import CrossEncoder
        if not hasattr(_try_rerank, "_model"):
            _try_rerank._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = _try_rerank._model.predict([(query, it.text) for it in items])
        ranked = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
        return [it for _, it in ranked[:top_k]]
    except Exception:
        return list(items)[:top_k]


# ══════════════════════════════════════════════════════════════
#  ANSWER GENERATION
# ══════════════════════════════════════════════════════════════

_RAG_SYSTEM = (
    "You are TruPharma Assistant, a medical drug-label information tool.\n"
    "Answer the question using ONLY the retrieved FDA drug-label evidence below.\n"
    "Cite every key claim with the bracket notation shown (e.g. [record_id::field]).\n"
    "Keep the answer concise (3-6 sentences). If the evidence is insufficient, "
    "respond exactly:\n"
    '"Not enough evidence in the retrieved context."'
    "\nDo NOT fabricate facts."
)


def _build_prompt(question: str, evidence: list) -> str:
    """Construct a RAG prompt with evidence citations."""
    lines = [f'{e["cite"]}  {e["content"]}' for e in evidence]
    block = "\n\n".join(lines)
    return (
        f"{_RAG_SYSTEM}\n\n"
        f"Evidence:\n{block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with citations):"
    )


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    """Call Google Gemini for grounded answer generation. Returns None on failure."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        if resp and resp.text:
            return resp.text.strip()
    except Exception as exc:
        warnings.warn(f"Gemini error: {exc}")
    return None


def _fallback_answer(question: str, evidence: list, n: int = 5) -> str:
    """
    Extractive fallback answer generator — no external LLM required.
    Selects the most relevant sentences from evidence and formats with citations.
    """
    if not evidence:
        return "Not enough evidence in the retrieved context."

    total = sum(len((e.get("content") or "").strip()) for e in evidence)
    if total < 200:
        return "Not enough evidence in the retrieved context."

    q_tok = set(tokenize(question))
    cands = []
    for e in evidence:
        cite = e["cite"]
        for sent in re.split(r"(?<=[.!?])\s+|\n+", (e.get("content") or "")):
            sent = sent.strip()
            if len(sent) < 30:
                continue
            s_tok = set(tokenize(sent))
            overlap = len(q_tok & s_tok)
            bonus = 2 if re.search(r"\d", sent) else 0
            cands.append((overlap + bonus, sent, cite))

    cands.sort(key=lambda x: x[0], reverse=True)
    picked, seen = [], set()
    for sc, sent, cite in cands:
        if sc <= 0:
            break
        key = sent[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(f"{sent} {cite}")
        if len(picked) >= n:
            break

    if not picked:
        return "Not enough evidence in the retrieved context."
    return "\n\n".join(picked)


def _confidence(evidence: list, answer: str) -> float:
    """Heuristic confidence score based on evidence coverage and citations."""
    if "Not enough evidence" in answer:
        return 0.0
    n = len(evidence)
    cites = len(re.findall(r"\[.*?\]", answer))
    return round(min(1.0, 0.30 + 0.08 * n + 0.04 * cites), 2)


# ══════════════════════════════════════════════════════════════
#  CSV LOGGING
# ══════════════════════════════════════════════════════════════

def _ensure_log():
    """Create the log directory and CSV header if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLS).writeheader()


def log_row(row: Dict[str, Any]):
    """Append one interaction row to the product metrics CSV."""
    _ensure_log()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        w.writerow({k: row.get(k, "") for k in LOG_COLS})


def read_logs(last_n: int = 20) -> List[Dict[str, str]]:
    """Read the most recent log rows for display."""
    if not LOG_CSV.exists():
        return []
    with open(LOG_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-last_n:]


# ══════════════════════════════════════════════════════════════
#  MAIN RAG PIPELINE
# ══════════════════════════════════════════════════════════════

def run_rag_query(
    query: str,
    *,
    gemini_key: str = "",
    method: str = "hybrid",
    top_k: int = 5,
    use_rerank: bool = False,
    api_limit: int = DEFAULT_LIMIT,
    max_records: int = DEFAULT_MAX_REC,
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:
      openFDA API fetch  ->  chunk  ->  index  ->  retrieve  ->  generate  ->  log

    Returns dict with: answer, evidence, latency_ms, confidence, num_records,
                       search_query, prompt, llm_used, method
    """
    t0 = time.time()

    # 1 ── Build openFDA search query from user text
    search_q = build_openfda_query(query, fields=FIELD_ALLOWLIST)

    # 2 ── Fetch + chunk + index (in-memory, no disk save)
    try:
        arts = build_artifacts(
            api_search=search_q,
            field_allowlist=FIELD_ALLOWLIST,
            field_blocklist=FIELD_BLOCKLIST,
            include_table_fields=False,
            min_chars=40,
            use_st=USE_SENTENCE_TRANSFORMERS,
            save=False,
            save_vectorizer=False,
            api_base_url=API_BASE,
            api_limit=api_limit,
            api_max_records=max_records,
            verbose=False,
        )
    except RuntimeError as exc:
        # API error (404 = no matching labels, etc.) — return graceful refusal
        lat = round((time.time() - t0) * 1000, 1)
        if "404" in str(exc) or "Not Found" in str(exc):
            err_answer = "Not enough evidence in the retrieved context."
        else:
            err_answer = f"Error fetching data from openFDA: {exc}"
        log_row({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": query[:200],
            "latency_ms": lat,
            "evidence_ids": "",
            "confidence": 0.0,
            "num_evidence": 0,
            "num_records": 0,
            "retrieval_method": method,
            "llm_used": False,
            "answer_preview": err_answer[:150],
        })
        return {
            "answer": err_answer,
            "evidence": [],
            "latency_ms": lat,
            "confidence": 0.0,
            "num_records": 0,
            "search_query": search_q,
            "prompt": "",
            "llm_used": False,
            "method": method,
        }

    corpus = arts["record_chunks"]
    index = arts["faiss_A"]
    bm25 = arts["bm25_A"]
    emb = (arts.get("manifest", {}).get("embedder") or {})
    e_type = emb.get("type")
    e_model = emb.get("model")
    vec = arts.get("vectorizer")
    n_recs = (arts.get("manifest", {}).get("counts") or {}).get("records", 0)

    # 3 ── Retrieve
    pool = max(20, top_k * 3)
    if method == "dense":
        items = [it for _, it in _dense(query, index, corpus, e_type, e_model, vec, pool)]
    elif method == "sparse":
        items = [it for _, it in _sparse(query, bm25, corpus, pool)]
    else:
        d = _dense(query, index, corpus, e_type, e_model, vec, pool)
        s = _sparse(query, bm25, corpus, pool)
        items = [it for _, it in _fuse(d, s, 0.5, pool)]

    # -- Free heavy indexing objects to keep memory low on cloud --
    del arts, index, bm25, corpus, vec
    gc.collect()

    # 4 ── Optional rerank
    if use_rerank and items:
        items = _try_rerank(query, items, top_k)
    else:
        items = items[:top_k]

    # 5 ── Build evidence pack
    evidence = [
        {
            "cite": f"[{it.chunk_id}]",
            "content": it.text[:1200],
            "doc_id": it.doc_id,
            "field": it.field,
        }
        for it in items
    ]

    # 6 ── Generate answer (Gemini LLM or extractive fallback)
    prompt = _build_prompt(query, evidence)
    llm_used = False
    answer = None

    if gemini_key:
        answer = _call_gemini(prompt, gemini_key)
        if answer:
            llm_used = True

    if answer is None:
        answer = _fallback_answer(query, evidence)

    # 7 ── Compute confidence
    conf = _confidence(evidence, answer)
    lat = round((time.time() - t0) * 1000, 1)

    # 8 ── Log interaction to CSV
    ev_ids = [e["cite"] for e in evidence]
    log_row({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": query[:200],
        "latency_ms": lat,
        "evidence_ids": "; ".join(ev_ids),
        "confidence": conf,
        "num_evidence": len(evidence),
        "num_records": n_recs,
        "retrieval_method": method,
        "llm_used": llm_used,
        "answer_preview": (answer or "")[:150],
    })

    return {
        "answer": answer,
        "evidence": evidence,
        "latency_ms": lat,
        "confidence": conf,
        "num_records": n_recs,
        "search_query": search_q,
        "prompt": prompt,
        "llm_used": llm_used,
        "method": method,
    }
