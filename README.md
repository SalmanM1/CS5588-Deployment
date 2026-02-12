# TruPharma GenAI Assistant

> **CS 5588 — Week 4 Capstone Module**
> Drug Label Evidence RAG System

**Team:** Salman Mirza, Amy Ngo, Nithin Songala

---

## Overview

TruPharma is a Retrieval-Augmented Generation (RAG) application that answers drug-label questions using official FDA data from the [openFDA Drug Label API](https://open.fda.gov/apis/drug/label/). The system fetches real-time drug labeling records, indexes them with hybrid retrieval (dense + sparse), and generates grounded answers with evidence citations.

### Target Users

| Persona | Example Task |
|---------|-------------|
| **Pharmacist** | "What dosage of acetaminophen is recommended and what are the warnings?" |
| **Clinician** | "What drug interactions should I know about for ibuprofen?" |
| **Patient** | "I take aspirin daily — when should I stop use?" |

### Value Proposition

Provides **faster time-to-answer** with **higher trust** by returning an evidence pack (drug label sections) and a **citation-enforced grounded answer**, refusing when evidence is insufficient.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit UI (Frontend)                │
│   Query Input  ·  Response  ·  Evidence  ·  Metrics/Logs │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  RAG Engine (rag_engine.py)               │
│                                                          │
│  1. Build openFDA search query from user text            │
│  2. Fetch drug label records via openFDA API             │
│  3. Chunk text fields (10 selected label sections)       │
│  4. Index: FAISS (dense) + BM25 (sparse)                │
│  5. Hybrid retrieval with reciprocal rank fusion         │
│  6. Generate answer (Gemini LLM or extractive fallback)  │
│  7. Log interaction to CSV                               │
└────────┬──────────────────────────┬──────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐    ┌───────────────────────────┐
│  openFDA API    │    │  Google Gemini 2.0 Flash   │
│  (Drug Labels)  │    │  (Optional LLM grounding)  │
└─────────────────┘    └───────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│               logs/product_metrics.csv                    │
│  timestamp · query · latency · evidence_ids · confidence  │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. User enters a drug-related question in the Streamlit UI
2. `rag_engine.py` converts the question into an openFDA API search query
3. Relevant drug label records are fetched in real-time from FDA servers
4. Text is chunked and indexed using dual retrieval (FAISS inner-product + BM25)
5. Top-K evidence is retrieved via hybrid fusion (dense + sparse)
6. A grounded answer is generated with citations (Gemini LLM or extractive fallback)
7. The interaction is logged to `logs/product_metrics.csv`
8. Results displayed: answer, evidence artifacts, latency metrics, and logs

### Selected Drug Label Fields (10)

| Field | Purpose |
|-------|---------|
| `active_ingredient` | Medicinal ingredients |
| `description` | Drug product overview |
| `dosage_and_administration` | Dosing guidance |
| `drug_interactions` | Drug/drug and drug/food interactions |
| `information_for_patients` | Patient safety info |
| `when_using` | Side effects and activity warnings |
| `overdosage` | Overdose symptoms and treatment |
| `stop_use` | When to stop and consult a doctor |
| `user_safety_warnings` | Hazard warnings |
| `warnings` | Serious adverse reactions |

---

## Deployed Application

**Live App:** [https://trupharm.streamlit.app/](https://trupharm.streamlit.app/)

---

## Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/SalmanM1/CS5588-Deployment.git
cd CS5588-Deployment

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run src/app/streamlit_app.py
```

### Optional: Gemini LLM

To use Google Gemini for answer generation instead of the extractive fallback:

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey)
2. Enter it in the app sidebar under **Advanced Settings > Gemini API key**

---

## Logging & Monitoring

All query interactions are logged to `logs/product_metrics.csv` with the following fields:

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp of the query |
| `query` | User's question (truncated to 200 chars) |
| `latency_ms` | End-to-end pipeline latency in milliseconds |
| `evidence_ids` | Chunk IDs of retrieved evidence |
| `confidence` | Heuristic confidence score (0–1) |
| `num_evidence` | Number of evidence items returned |
| `num_records` | Drug label records fetched from FDA API |
| `retrieval_method` | hybrid / dense / sparse |
| `llm_used` | Whether Gemini LLM was used |
| `answer_preview` | First 150 chars of the generated answer |

---

## Production Failure Scenario & Mitigation

**Scenario:** openFDA API returns 0 results for an obscure or misspelled drug name.

**Mitigation:**
- The system detects empty result sets and returns a clear "Not enough evidence" message rather than hallucinating
- Logging captures the failed query for later analysis
- Future improvement: add fuzzy drug-name matching and spell-check suggestions before querying the API

---

## Deployment & Scaling

| Aspect | Approach |
|--------|----------|
| **Hosting** | Streamlit Community Cloud (free tier) |
| **Data** | Real-time openFDA API (no local data storage needed) |
| **Scaling** | API rate limits managed via pagination; add API key for higher limits |
| **Monitoring** | CSV-based logging; extend to cloud logging (e.g., CloudWatch) for production |
| **CI/CD** | GitHub integration with Streamlit Cloud for auto-deploy on push |

---

## Repository Structure

```
Week-4-Assignment--main/
├── data/                          # Data directory (placeholder)
├── logs/
│   └── product_metrics.csv        # Interaction logs (≥5 records)
├── src/
│   ├── openfda_rag.py             # openFDA API fetching, chunking, indexing
│   ├── rag_engine.py              # RAG pipeline: retrieve → generate → log
│   ├── Week 4.ipynb               # Development notebook
│   └── app/
│       ├── .streamlit/config.toml # Streamlit theme config
│       ├── streamlit_app.py       # Main app (Primary Demo)
│       └── pages/
│           └── stress_test.py     # Stress test / scenario validation
├── requirements.txt
└── README.md
```

---

## Impact Evaluation

- **Workflow improvement:** Reduces manual label scanning from 10–15 min to under 30 sec per question
- **Time-to-decision:** Estimated 80% reduction in time-to-answer for drug-label queries
- **Trust indicators:** Every answer includes evidence chunk IDs, source fields, and confidence scores; system refuses to answer when evidence is insufficient

---

*CS 5588 · Spring 2026 · Week 4 Assignment*
