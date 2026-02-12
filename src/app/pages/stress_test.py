import streamlit as st
import time
from datetime import datetime

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Stress Test | Scenario Validation",
    page_icon="⚠️",
    layout="wide"
)

# -----------------------------------
# HIDE STREAMLIT BUILT-IN PAGE NAV
# (file-name buttons)
# -----------------------------------
st.markdown("""
<style>
div[data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] nav { display: none !important; }
section[data-testid="stSidebar"] ul[role="list"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }  
section[data-testid="stSidebar"] button:has(svg) { display:none !important; }      
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# STYLES: MATCH THE COMPARISON VIEW
# -----------------------------------
st.markdown("""
<style>
.main-header-bar {
    background: linear-gradient(90deg, #F2994A, #EB5757);
    color: white;
    padding: 12px 16px;
    border-radius: 10px;
    font-weight: 600;
    margin-bottom: 14px;
}

/* Page title look */
.page-title {
  font-size: 34px;
  font-weight: 800;
  margin-bottom: 4px;
}
.page-subtitle {
  color: #6b7280;
  font-weight: 600;
  margin-bottom: 14px;
}

/* Column containers */
.panel {
  border-radius: 16px;
  padding: 0;
  border: 1px solid #E5E7EB;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  background: #fff;
}

/* Panel headers */
.panel-header {
  padding: 12px 18px;
  font-weight: 900;
  font-size: 18px;
  color: #111827;
}
.panel-subheader {
  padding: 0 18px 12px 18px;
  font-weight: 700;
  color: #4b5563;
}

.panel-header.primary { background: #CFE7C8; }
.panel-header.stress  { background: #F7C08A; }

/* ✅ Make scenario header highlights pill-shaped */
.panel-header.primary,
.panel-header.stress {
  border-radius: 16px !important;
  margin: 14px 14px 6px 14px !important;
  width: calc(100% - 28px) !important;
}

/* ✅ Heading pill (gray label) */
.section-pill {
  display: inline-block;
  background: #F3F4F6;
  border: 1px solid #E5E7EB;
  color: #111827;
  border-radius: 16px;
  padding: 8px 14px;
  font-weight: 900;
  font-size: 15px;
  margin: 10px 0 8px 0;
}

/* Inner blocks become “content areas” (no extra card box) */
.inner-card {
  margin: 10px 18px;
  border: none;
  background: transparent;
  padding: 0;
}

.mini {
  color: #4b5563;
  font-weight: 600;
}

/* Bullet list */
.bullets {
  margin: 0;
  padding-left: 18px;
}
.bullets li { margin: 6px 0; }

/* Bottom criteria boxes */
.criteria {
  border-radius: 16px;
  border: 1px solid #E5E7EB;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  background: #fff;
}

.criteria-header {
  padding: 10px 14px;
  font-weight: 900;
  font-size: 18px;
  color: #111827;
}

.criteria-header.success { background: #CFE7C8; }
.criteria-header.pass    { background: #F7C08A; }

.criteria-body {
  padding: 12px 14px;
  background: #F9FAFB;
  font-weight: 600;
  color: #374151;
}

/* Sidebar nav styling */
.sidebar-title {
  font-size: 15px;
  font-weight: 800;
  margin: 10px 0 8px 0;
}
.scenario-card {
  padding: 10px 12px;
  border-radius: 10px;
  margin-bottom: 8px;
  font-weight: 700;
  line-height: 1.2;
}
.primary-active {
  background-color: #E8F5E9;
  border-left: 6px solid #2E7D32;
}
.stress-active {
  background-color: #FFF3E0;
  border-left: 6px solid #EF6C00;
}

/* Global font override */
html, body, * {
    font-family: "Times New Roman", Times, serif !important;
    line-height: 1.4;
}

                       
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# CUSTOM SIDEBAR NAV (Primary vs Stress)
# -----------------------------------
st.sidebar.markdown("<div class='sidebar-title'>Scenario Mode</div>", unsafe_allow_html=True)

if st.sidebar.button("⬅ Return to Primary Demo", key="go_primary"):
    st.switch_page("streamlit_app.py")

st.sidebar.markdown(
    "<div class='scenario-card stress-active'>🟠 Stress Test<br><small>Edge case / robustness validation</small></div>",
    unsafe_allow_html=True
)

# Optional stress controls
st.sidebar.markdown("---")
stress_condition = st.sidebar.radio(
    "Stress Condition (choose one)",
    ["Rare input", "Large doc", "Heavy traffic", "Conflicting evidence"]
)
run = st.sidebar.button("Run Stress Test")

# -----------------------------------
# SESSION STATE (shared across pages)
# -----------------------------------
if "primary_last_run" not in st.session_state:
    st.session_state.primary_last_run = {
        "query": 'drug="Ibuprofen", symptom="dizziness"',
        "confidence": "89%",
        "evidence_count": 3
    }

if "stress_last_run" not in st.session_state:
    st.session_state.stress_last_run = {
        "condition": None,
        "latency_ms": 0,
        "errors": "None",
        "evidence_ids": [],
        "config": "top_k=2 (auto-reduced)"
    }

# -----------------------------------
# RUN STRESS TEST (placeholder behavior)
# -----------------------------------
if run:
    t0 = time.time()
    simulated_delay = {
        "Rare input": 1.1,
        "Large doc": 2.2,
        "Heavy traffic": 3.0,
        "Conflicting evidence": 1.7
    }[stress_condition]

    time.sleep(simulated_delay)
    latency = round((time.time() - t0) * 1000, 2)

    if stress_condition == "Heavy traffic":
        config = "fallback model • limit context • retry top_k=2"
        errors = "None (degraded gracefully)"
    elif stress_condition == "Large doc":
        config = "limit context • chunk cap • retry top_k=2"
        errors = "None (context trimmed)"
    elif stress_condition == "Conflicting evidence":
        config = "rerank evidence • show uncertainty • top_k=3→2"
        errors = "None (conflict flagged)"
    else:
        config = "fallback retrieval • retry top_k=2"
        errors = "None"

    st.session_state.stress_last_run = {
        "condition": stress_condition,
        "latency_ms": latency,
        "errors": errors,
        "evidence_ids": ["E102", "E331"],
        "config": config
    }

# -----------------------------------
# PAGE HEADER
# -----------------------------------
st.markdown("<div class='page-title'>Scenario Validation View</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'></div>", unsafe_allow_html=True)

st.markdown(
    "<div class='main-header-bar'>Stress Test: TruPharma RAG vs Edge Case Scenarios</div>",
    unsafe_allow_html=True
)

# -----------------------------------
# TWO SIDE-BY-SIDE PANELS
# -----------------------------------
left, right = st.columns(2, gap="large")

# ---- Primary Demo Scenario panel ----
with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header primary'>Primary Demo Scenario</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-subheader'>Normal user workflow</div>", unsafe_allow_html=True)

    st.markdown("<div class='inner-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>Input</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mini'>Example: {st.session_state.primary_last_run['query']}<br>(or campus question / weather query)</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='inner-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>Expected Output</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mini'>Verified answer + short explanation<br>"
        f"Confidence indicator: {st.session_state.primary_last_run['confidence']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='inner-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>Evidence</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mini'>Citations / retrieved chunks / artifacts<br>"
        f"Evidence count: {st.session_state.primary_last_run['evidence_count']}</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Stress-Test Scenario panel ----
with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header stress'>Stress-Test Scenario</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-subheader'>Edge case / robustness check</div>", unsafe_allow_html=True)

    st.markdown("<div class='inner-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>Stress Condition</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini'>Choose ONE:</div>", unsafe_allow_html=True)
    st.markdown("""
    <ul class="bullets">
      <li>rare input</li>
      <li>large doc</li>
      <li>heavy traffic</li>
      <li>conflicting evidence</li>
    </ul>
    """, unsafe_allow_html=True)
    if st.session_state.stress_last_run["condition"]:
        st.markdown(f"<div class='mini'><b>Selected:</b> {st.session_state.stress_last_run['condition']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='inner-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>System Behavior</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini'>Graceful degradation:</div>", unsafe_allow_html=True)
    st.markdown("""
    <ul class="bullets">
      <li>fallback model</li>
      <li>limit context</li>
      <li>retry with smaller top-k</li>
    </ul>
    """, unsafe_allow_html=True)
    if st.session_state.stress_last_run["condition"]:
        st.markdown(f"<div class='mini'><b>Applied:</b> {st.session_state.stress_last_run['config']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='inner-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>Monitoring / Logs</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini'>Must show logs:<br>latency • errors • evidence IDs • config</div>", unsafe_allow_html=True)

    if st.session_state.stress_last_run["condition"]:
        st.markdown(f"- **Latency (ms):** {st.session_state.stress_last_run['latency_ms']}")
        st.markdown(f"- **Errors:** {st.session_state.stress_last_run['errors']}")
        st.markdown(f"- **Evidence IDs:** {', '.join(st.session_state.stress_last_run['evidence_ids'])}")
        st.markdown(f"- **Config:** {st.session_state.stress_last_run['config']}")
    else:
        st.info("Run a stress test to populate logs.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# SUCCESS / PASS CRITERIA (BOTTOM)
# -----------------------------------
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("<div class='criteria'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header primary'>Success Criteria</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='criteria-body'>
      ✅ Week 4 requirement: demonstrate BOTH scenarios and pass criteria in submission.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='criteria'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header stress'>Pass Criteria</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='criteria-body'>
      ✅ Meets expectations in both scenarios.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
