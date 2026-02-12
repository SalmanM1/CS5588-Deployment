import streamlit as st
from datetime import datetime
import time

st.set_page_config(page_title="Primary Demo | Healthcare RAG", page_icon="🩺", layout="wide")

# -------------------------------
# Hide Streamlit built-in page nav
# -------------------------------
st.markdown("""
<style>
div[data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] nav { display: none !important; }
section[data-testid="stSidebar"] ul[role="list"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; }    
section[data-testid="stSidebar"] button:has(svg) { display:none !important; }    
</style>
""", unsafe_allow_html=True)

# -------------------------------
# App styling (cards, header, etc.)
# -------------------------------
st.markdown(""" <style> .main-header-bar { background: linear-gradient(90deg, #F2994A, #EB5757); color: white; padding: 12px 16px; border-radius: 10px; font-weight: 600; margin-bottom: 14px; } .scenario-card { padding: 10px 12px; border-radius: 10px; margin-bottom: 8px; font-weight: 700; line-height: 1.2; } .primary-active { background-color: #E8F5E9; border-left: 6px solid #2E7D32; } .card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); margin-bottom: 14px; } .card-title { font-weight: 800; font-size: 16px; margin-bottom: 8px; } .card-title.response { color: #1f7a8c; } .card-title.evidence { color: #d35400; } .card-title.metrics { color: #2e7d32; } .card-title.logs { color: #6b7280; } .bullets { margin: 0; padding-left: 18px; } .bullets li { margin: 6px 0; } /* Pill row */ .pillbtn > button { border-radius: 999px !important; padding: 0.6rem 1rem !important; font-weight: 700 !important; } .pill-row { display: flex; gap: 14px; margin: 10px 0 18px 0; } .pill-link { flex: 1; text-align: center; padding: 14px 14px; border-radius: 14px; border: 1px solid #d1d5db; background: #ffffff; font-weight: 800; color: #111827; text-decoration: none !important; box-shadow: 0 1px 2px rgba(0,0,0,0.06); } .pill-link:hover { background: #f3f4f6; } .pill-link.active { background: #5DADE2; border-color: #4b93c9; color: white !important; } .pill-link.active.response { background: #5DADE2; border-color: #4b93c9; } .pill-link.active.evidence { background: #F2994A; border-color: #E67E22; } .pill-link.active.metrics { background: #6FCF97; border-color: #27AE60; } .pill-link.active.logs { background: #BDBDBD; border-color: #9CA3AF; color: #111827 !important; } /* Global font override */ html, body, * { font-family: "Times New Roman", Times, serif !important; line-height: 1.4; } </style> """, unsafe_allow_html=True)





# -------------------------------
# Session state init
# -------------------------------
if "active_panel" not in st.session_state:
    st.session_state.active_panel = "ALL"  # ALL = overall dashboard

if "response" not in st.session_state:
    st.session_state.response = None
if "evidence" not in st.session_state:
    st.session_state.evidence = []
if "latency" not in st.session_state:
    st.session_state.latency = 0
if "logs" not in st.session_state:
    st.session_state.logs = []

def toggle_panel(name: str):
    """Click pill to focus; click again to return to ALL."""
    st.session_state.active_panel = "ALL" if st.session_state.active_panel == name else name

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("Scenario Mode")
st.sidebar.markdown(
    "<div class='scenario-card primary-active'>🟢 Primary Demo<br><small>Normal user workflow</small></div>",
    unsafe_allow_html=True
)
if st.sidebar.button("⚠️ Go to Stress Test", key="go_stress"):
    st.switch_page("pages/stress_test.py")


st.sidebar.title("Use Case Input")
query_text = st.sidebar.text_area("Query Text", placeholder="Enter clinical question...")

run = st.sidebar.button("Run")


st.sidebar.title("Reset Session")
if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.rerun()

# -------------------------------
# Run logic FIRST (so render uses updated state)
# -------------------------------
if run and query_text:
    t0 = time.time()
    time.sleep(0.5)

    st.session_state.evidence = [
        ("E101", "ADA guideline excerpt...", "download (TBD)"),
        ("E233", "Peer-reviewed evidence snippet...", "download (TBD)"),
        ("E331", "Hospital protocol reference...", "download (TBD)"),
    ]

    st.session_state.response = {
        "short": "Ibuprofen may cause dizziness in some patients.",
        "confidence": "89%",
        "reasoning": "Retrieved clinical sources list dizziness as a known side effect.",
        "next_steps": "Review dosage, check interactions, monitor symptoms."
    }

    st.session_state.latency = round((time.time() - t0) * 1000, 2)
    st.session_state.logs.append(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Run completed in {st.session_state.latency} ms"
    )

    # For your comparison page
    st.session_state.primary_last_run = {
        "query": query_text,
        "confidence": st.session_state.response["confidence"],
        "evidence_count": len(st.session_state.evidence)
    }

# -------------------------------
# Main Panel Header
# -------------------------------
st.markdown("## TruPharma GenAI Assistant")

project_name = "TruPharma GenAI Assistant"
status = "Primary Demo"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown(
    f"<div class='main-header-bar'>Prototype Primary Demo</div>",
    unsafe_allow_html=True
)

# -------------------------------
# ONE pill row (selectable)
# -------------------------------
# --- state init (put once, near your other session_state init) ---
if "active_panel" not in st.session_state:
    st.session_state.active_panel = "ALL"

def set_panel(name: str):
    # click same pill again => back to ALL
    st.session_state.active_panel = "ALL" if st.session_state.active_panel == name else name


# --- ONE pill row (put under your header) ---
c1, c2, c3, c4 = st.columns(4, gap="small")

c1.button(
    "Response",
    use_container_width=True,
    type="primary" if st.session_state.active_panel == "Response" else "secondary",
    key="pill_response",
    on_click=set_panel,
    args=("Response",),
)

c2.button(
    "Evidence / Artifacts",
    use_container_width=True,
    type="primary" if st.session_state.active_panel == "Evidence" else "secondary",
    key="pill_evidence",
    on_click=set_panel,
    args=("Evidence",),
)

c3.button(
    "Metrics & Monitoring",
    use_container_width=True,
    type="primary" if st.session_state.active_panel == "Metrics" else "secondary",
    key="pill_metrics",
    on_click=set_panel,
    args=("Metrics",),
)

c4.button(
    "Logs",
    use_container_width=True,
    type="primary" if st.session_state.active_panel == "Logs" else "secondary",
    key="pill_logs",
    on_click=set_panel,
    args=("Logs",),
)

st.caption("Tip: Click the same pill again to return to the full dashboard view.")



# -------------------------------
# Render helpers
# -------------------------------
def render_response_only():
    st.markdown("<div class='card'><div class='card-title response'>Response Panel</div>", unsafe_allow_html=True)

    if not st.session_state.response:
        st.markdown("""
        <ul class="bullets">
          <li>Short Answer + Confidence</li>
          <li>Reasoning Summary</li>
          <li>Actionable Next Steps</li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state.response
        st.markdown(f"**Short Answer:** {r['short']}")
        st.markdown(f"**Confidence:** {r['confidence']}")
        st.markdown("**Reasoning Summary:**")
        st.write(r["reasoning"])
        st.markdown("**Actionable Next Steps:**")
        st.write(r["next_steps"])

    st.markdown("</div>", unsafe_allow_html=True)


def render_evidence_only():
    st.markdown("<div class='card'><div class='card-title evidence'>Evidence / Artifacts</div>", unsafe_allow_html=True)

    if not st.session_state.evidence:
        st.markdown("""
        <ul class="bullets">
          <li>Evidence ID</li>
          <li>Source Text</li>
          <li>Download Link</li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        for evid, text, dl in st.session_state.evidence:
            st.markdown(f"**{evid}** — {text}")
            st.caption(f"Download: {dl}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_metrics_only():
    st.markdown("<div class='card'><div class='card-title metrics'>Metrics & Monitoring</div>", unsafe_allow_html=True)

    st.markdown(f"- **Latency (ms):** {st.session_state.latency}")
    st.markdown(f"- **Evidence Count:** {len(st.session_state.evidence)}")
    st.markdown("- **Errors / Fallbacks:** None")

    st.markdown("</div>", unsafe_allow_html=True)


def render_logs_only():
    st.markdown("<div class='card'><div class='card-title logs'>Logs</div>", unsafe_allow_html=True)

    if not st.session_state.logs:
        st.write("No logs yet.")
    else:
        for line in st.session_state.logs[-10:][::-1]:
            st.write(line)

    st.markdown("</div>", unsafe_allow_html=True)

def render_overall_panel():
    left, right = st.columns([2.2, 1.2], gap="large")

    with left:
        render_response_only()

    with right:
        render_evidence_only()
        render_metrics_only()

    render_logs_only()

# -------------------------------
# Conditional views
# -------------------------------
active = st.session_state.active_panel

if active == "ALL":
    render_overall_panel()
elif active == "Response":
    render_response_only()
elif active == "Evidence":
    render_evidence_only()
elif active == "Metrics":
    render_metrics_only()
elif active == "Logs":
    render_logs_only()
