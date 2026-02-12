# Week-4 Integration Report
## TruPharma — Drug Label Evidence RAG System

**Course:** CS 5588 · Spring 2026 · Week 4 Assignment
**Team:** Salman Mirza, Amy Ngo, Nithin Songala
**Module:** RAG LLM (Salman Mirza)
**Deployed App:** [https://trupharm.streamlit.app/](https://trupharm.streamlit.app/)
**Repository:** [https://github.com/SalmanM1/CS5588-Deployment](https://github.com/SalmanM1/CS5588-Deployment)

---

## 1. Where the Module Fits in the Capstone Architecture

The RAG LLM module is the core intelligence layer of the TruPharma system. It sits between the user-facing Streamlit UI and the external data source (openFDA Drug Label API), orchestrating the full pipeline from question to grounded answer:

```
User ──► Streamlit UI ──► RAG Engine ──► openFDA API
                              │
                              ├──► Chunking & Indexing (FAISS + BM25)
                              ├──► Hybrid Retrieval (dense + sparse fusion)
                              ├──► Answer Generation (Gemini LLM / extractive fallback)
                              └──► Logging (product_metrics.csv)
```

The module receives natural-language drug questions, converts them into API search queries, fetches real-time FDA drug label records, chunks and indexes the text across 10 selected label fields, retrieves the most relevant evidence using hybrid search (reciprocal rank fusion of FAISS inner-product and BM25 scores), and generates a citation-enforced answer. If a Gemini API key is provided, the system uses Google Gemini 2.0 Flash for LLM-grounded generation; otherwise, it falls back to an extractive method that selects and concatenates the highest-scoring evidence passages.

---

## 2. Supported User Workflow

| Step | User Action | System Response |
|------|-------------|-----------------|
| 1 | Opens the app at [trupharm.streamlit.app](https://trupharm.streamlit.app/) | Displays the query interface with example questions |
| 2 | Types a drug-related question (e.g., "What are the drug interactions for ibuprofen?") | Converts question to openFDA search, fetches records |
| 3 | Clicks **Search** | Runs hybrid retrieval, generates grounded answer |
| 4 | Reviews the **Response Panel** | Sees the answer with inline citation IDs (e.g., `[doc_id::field]`) |
| 5 | Expands the **Evidence Panel** | Views the source text chunks, field names, and confidence scores |
| 6 | Checks the **Metrics Panel** | Sees latency, number of records fetched, retrieval method, and confidence |
| 7 | Views the **Log History** tab | Reviews past interactions logged to CSV |

The **Stress Test** page (accessible via sidebar navigation) runs three pre-defined scenarios automatically — a drug interaction query, a dosage/warnings query, and an out-of-scope refusal test — to validate pipeline correctness and measure latency.

---

## 3. Application Interface

The Streamlit application has two pages:

- **Primary Demo** (`streamlit_app.py`): Main query interface with a two-column layout — left column for query input and response, right column for evidence artifacts and pipeline metrics.
- **Stress Test** (`pages/stress_test.py`): Automated scenario validation that runs three test queries in sequence, displaying pass/fail results, latency measurements, and evidence summaries.

*(See application screenshots attached separately or visit the live app.)*

---

## 4. Logging Example

All interactions are logged to `logs/product_metrics.csv`. Below are sample rows demonstrating the tracked metrics:

| timestamp | query | latency_ms | confidence | num_evidence | retrieval_method | llm_used |
|-----------|-------|-----------|------------|-------------|-----------------|----------|
| 2026-02-10T14:23:15Z | What are the drug interactions for ibuprofen? | 4523.2 | 0.78 | 5 | hybrid | False |
| 2026-02-10T15:01:42Z | Recommended dosage for acetaminophen and warnings? | 3891.7 | 0.82 | 5 | hybrid | False |
| 2026-02-10T16:15:33Z | Safety warnings for caffeine-containing products? | 5102.4 | 0.74 | 4 | hybrid | False |
| 2026-02-11T09:45:21Z | Warnings for aspirin use during pregnancy? | 4201.8 | 0.80 | 5 | hybrid | False |
| 2026-02-11T10:30:55Z | Overdosage symptoms for diphenhydramine? | 3654.1 | 0.76 | 4 | hybrid | False |
| 2026-02-11T14:12:08Z | Projected cost of antimicrobial resistance to GDP in 2050? | 2103.5 | 0.00 | 3 | hybrid | False |
| 2026-02-12T08:05:44Z | Aspirin overdosage and when to stop use? | 5891.3 | 0.82 | 5 | hybrid | False |

**Key observations:**
- Average latency is **2–5 seconds** per query after optimization (down from 10+ minutes before caching and TF-IDF were introduced).
- The out-of-scope question (antimicrobial resistance GDP cost) correctly returns **confidence = 0.0** and the refusal message "Not enough evidence in the retrieved context."
- The log currently contains **20 interaction records**, well exceeding the ≥5 requirement.

---

## 5. Production Failure Scenario & Mitigation

**Scenario:** The openFDA API returns 0 results for an obscure, misspelled, or non-drug query (e.g., "What is the projected cost of antimicrobial resistance to GDP in 2050?").

**What happens without mitigation:** The pipeline would have no documents to index, potentially causing a crash or, worse, generating a hallucinated answer with no supporting evidence.

**Implemented mitigation:**
1. **Empty result detection:** If the API returns 0 records or a 404 error, the system immediately returns "Not enough evidence in the retrieved context." instead of attempting retrieval.
2. **Confidence scoring:** The heuristic confidence score drops to **0.0** when no evidence chunks are relevant, providing a clear trust signal to the user.
3. **Logging:** Failed queries are logged with `confidence=0.0` and empty `evidence_ids`, enabling post-hoc analysis of query gaps.
4. **Graceful UI handling:** The Streamlit frontend displays the refusal message in the response panel and shows "No evidence found" in the evidence panel, rather than crashing.

**Future improvement:** Add fuzzy drug-name matching and spell-check suggestions before querying the API, reducing the number of zero-result queries caused by typos.

---

## 6. Deployment Readiness Plan

### Architecture
See the architecture diagram in the README. The system uses a three-tier design: Streamlit UI → RAG Engine → External APIs (openFDA + optional Gemini LLM).

### Data Flow & Logging
1. User query enters through Streamlit → `rag_engine.run_rag_query()` is called
2. Query is transformed into an openFDA API search string
3. Up to 50 drug label records are fetched, chunked (250-word windows with 40-word overlap), and indexed
4. Hybrid retrieval (FAISS + BM25 with reciprocal rank fusion) returns top-K evidence
5. Answer is generated with inline citations
6. Full interaction (timestamp, query, latency, evidence IDs, confidence, etc.) is appended to `logs/product_metrics.csv`

### Hosting & Scaling

| Aspect | Current | Production Path |
|--------|---------|-----------------|
| **Hosting** | Streamlit Community Cloud (free) | Streamlit Cloud or containerized on AWS/GCP |
| **Data** | Real-time openFDA API (no local storage) | Add Redis caching for frequently queried drugs |
| **Scaling** | Single instance, API pagination | Horizontal scaling with load balancer; API key for higher rate limits |
| **Monitoring** | CSV logging | Cloud logging (CloudWatch / Stackdriver) + alerting |
| **CI/CD** | GitHub → Streamlit auto-deploy on push | Add GitHub Actions for testing + automated deployment |

---

## 7. Impact Evaluation

| Metric | Before (Manual) | After (TruPharma) | Improvement |
|--------|-----------------|-------------------|-------------|
| **Time-to-answer** | 10–15 min (scanning label PDFs) | < 5 sec per query | ~99% reduction |
| **Citation coverage** | Manual copy-paste of label sections | Automatic inline citations with chunk IDs | Full traceability |
| **Refusal accuracy** | User may overlook missing info | System refuses when confidence = 0 | Prevents misinformation |
| **Trust indicators** | None | Confidence score, evidence IDs, source fields displayed | Transparent decision basis |

**Workflow improvement:** Pharmacists, clinicians, and regulatory analysts no longer need to manually scan lengthy drug label PDFs. The RAG system retrieves relevant label sections in seconds and produces answers that cite exactly which label field and document the information came from.

**Estimated time-to-decision improvement:** ~99% reduction — from 10–15 minutes of manual searching to under 5 seconds of automated retrieval and answer generation.

**Trust indicators:** Every answer includes (1) inline citation IDs linking to specific label sections, (2) a heuristic confidence score (0–1), (3) the number of evidence chunks retrieved, and (4) a clear refusal message when evidence is insufficient. These indicators allow users to verify answers against source material and trust the system's outputs for clinical and compliance decisions.

---

*CS 5588 · Spring 2026 · Week 4 Integration Report*
