"""
Streamlit frontend for LegalLens — Classy & Modern Legal AI Review Assistant.
Calls the FastAPI backend over HTTP.

Run backend first: uvicorn main:app --reload
Then run Streamlit: streamlit run streamlit_app.py
"""
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="LegalLens AI — Legal Document Review",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS: Modern, Classy Dark Luxe Theme ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 80% 10%, #0F1714 0%, #090D0B 50%, #050706 100%);
        color: #E2E8F0;
    }

    /* Top Navigation Header */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: rgba(18, 24, 21, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 24px;
    }

    .brand-logo-container {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon-box {
        background: linear-gradient(135deg, #10B981 0%, #047857 100%);
        width: 46px;
        height: 46px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.35);
    }

    .brand-text-main {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .brand-sub-tag {
        font-size: 11px;
        color: #10B981;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    /* Cards & Containers */
    .classy-card {
        background: rgba(18, 24, 21, 0.65);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    .eyebrow {
        color: #10B981;
        font-size: 11px;
        letter-spacing: 2.5px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .hero-sub {
        color: #94A3B8;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    /* Pipeline Step Cards */
    .pipeline-card {
        background: rgba(22, 30, 26, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }

    .pipeline-card.waiting {
        border-color: rgba(255, 255, 255, 0.06);
        opacity: 0.7;
    }

    .pipeline-card.running {
        border-color: #34D399;
        background: rgba(16, 185, 129, 0.1);
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.2);
    }

    .pipeline-card.done {
        border-color: rgba(52, 211, 153, 0.4);
        background: rgba(16, 185, 129, 0.04);
    }

    .step-num {
        font-family: 'JetBrains Mono', monospace;
        color: #10B981;
        font-weight: 700;
        font-size: 12px;
    }

    .step-title {
        color: #F8FAFC;
        font-weight: 700;
        font-size: 15px;
        margin-left: 8px;
    }

    .step-desc {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 4px;
        line-height: 1.4;
    }

    .status-badge {
        float: right;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }

    .status-badge.waiting {
        background: rgba(255, 255, 255, 0.05);
        color: #64748B;
    }

    .status-badge.running {
        background: rgba(52, 211, 153, 0.2);
        color: #34D399;
        animation: pulse 1.5s infinite;
    }

    .status-badge.done {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
    }

    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }

    /* Widget Styling */
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background-color: rgba(15, 23, 20, 0.85) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
    }

    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.25) !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
        width: 100%;
    }

    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
    }

    /* Document Status Chip */
    .doc-info-bar {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .doc-id-code {
        font-family: 'JetBrains Mono', monospace;
        color: #34D399;
        font-size: 12px;
        background: rgba(0, 0, 0, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
    }

    .reviewer-banner {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(180, 83, 9, 0.08));
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 14px;
        padding: 22px;
        margin-top: 20px;
        margin-bottom: 16px;
    }

    .answer-card {
        background: rgba(15, 23, 20, 0.9);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def check_backend_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


# ---- Session State Setup ----
for key, default in {
    "document_id": None,
    "document_path": None,
    "document_name": None,
    "pending_thread_id": None,
    "pending_draft": None,
    "pending_critique": None,
    "query_input": "",
    "final_answer": None,
    "history": [],
    "step_status": {
        "retrieve": "waiting",
        "draft": "waiting",
        "critique": "waiting",
        "human_review": "waiting",
        "finalize": "waiting",
    },
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

PIPELINE_STEPS = [
    ("retrieve", "Hybrid Retrieval", "Extracts clause context via Chroma vector search & BM25 keyword matching"),
    ("draft", "Grounding & Draft", "Generates an initial answer backed strictly by contract clauses"),
    ("critique", "Self-Correction & Critique", "Fact-checks claims and verifies context fidelity"),
    ("human_review", "Human Sign-off Desk", "Expert review checkpoint for judgment calls and risk flags"),
    ("finalize", "Final Audit Report", "Finalizes verified legal analysis with source grounding"),
]


def reset_pipeline_status():
    st.session_state.step_status = {k: "waiting" for k, _, _ in PIPELINE_STEPS}


def set_query(text: str):
    st.session_state.query_input = text


# ---- Top Navigation Header ----
api_online = check_backend_api()
status_html = (
    '<div class="status-pill status-online">● Backend API Connected</div>'
    if api_online
    else '<div class="status-pill status-offline">● Backend Offline (uvicorn main:app)</div>'
)

st.markdown(f"""
<div class="brand-header">
    <div class="brand-logo-container">
        <div class="brand-icon-box">⚖️</div>
        <div>
            <div class="brand-sub-tag">Legal AI Workspace</div>
            <div class="brand-text-main">LegalLens AI</div>
        </div>
    </div>
    <div>
        {status_html}
    </div>
</div>
""", unsafe_allow_html=True)

if not api_online:
    st.warning(
        "⚠️ **FastAPI Backend is unreachable.** Please launch the server in your terminal:\n\n"
        "```bash\n"
        "uvicorn main:app --reload\n"
        "```"
    )

# ---- Main Layout ----
left_col, right_col = st.columns([1.35, 1])

with left_col:
    # ---- 1. Document Indexing Section ----
    st.markdown('<div class="eyebrow">DOCUMENT WORKSPACE</div>', unsafe_allow_html=True)

    if st.session_state.document_id is None:
        st.markdown(
            '<div class="hero-sub">Upload a legal contract or lease agreement (PDF/TXT) to initialize hybrid indexing and automated clause audit.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload legal contract",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.button("⚡ Index & Process Document"):
                with st.spinner("Indexing document chunks with hybrid vector store..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        response = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.document_id = data["document_id"]
                            st.session_state.document_path = data["document_path"]
                            st.session_state.document_name = uploaded_file.name
                            st.success(f"Successfully indexed '{uploaded_file.name}'")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"Could not connect to API server: {e}")
    else:
        # Show active document badge bar
        st.markdown(f"""
        <div class="doc-info-bar">
            <div>
                <span style="font-weight: 700; color: #F8FAFC; font-size: 15px;">📄 {st.session_state.document_name or 'Active Contract'}</span>
                <span style="margin-left: 12px;" class="doc-id-code">ID: {st.session_state.document_id[:12]}...</span>
            </div>
            <div>
                <span style="color: #34D399; font-size: 12px; font-weight: 600;">✓ Indexed & Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("🔄 Change File"):
                st.session_state.document_id = None
                st.session_state.document_path = None
                st.session_state.document_name = None
                st.session_state.final_answer = None
                st.session_state.pending_thread_id = None
                reset_pipeline_status()
                st.rerun()

    # ---- 2. Question & Analysis Studio ----
    if st.session_state.document_id:
        st.markdown('<div class="eyebrow" style="margin-top: 16px;">CLAUSE AUDIT & QUERY STUDIO</div>', unsafe_allow_html=True)

        st.text_input(
            "Legal Query",
            key="query_input",
            placeholder="Ask a question (e.g. Is the security deposit clause risky for the tenant?)",
            label_visibility="collapsed",
        )

        # Quick Suggestion Chips
        st.caption("Suggested Legal Inquiries:")
        chip_cols = st.columns(3)
        suggestions = [
            "What is the monthly rent?",
            "Is this deposit clause risky?",
            "What is the notice period?",
        ]

        for col, sug in zip(chip_cols, suggestions):
            col.button(sug, key=f"sug_{sug}", on_click=set_query, args=(sug,))

        st.write("")
        if st.button("🔍 Run Legal Review Pipeline"):
            if not st.session_state.query_input.strip():
                st.warning("Please enter a question or click a suggestion.")
            else:
                active_query = st.session_state.query_input.strip()
                reset_pipeline_status()
                st.session_state.step_status["retrieve"] = "running"
                st.session_state.final_answer = None
                st.session_state.pending_thread_id = None

                with st.spinner("Executing RAG Pipeline & Self-Critique..."):
                    payload = {
                        "document_id": st.session_state.document_id,
                        "document_path": st.session_state.document_path,
                        "query": active_query,
                    }
                    try:
                        res = requests.post(f"{API_BASE}/review", json=payload, timeout=90)
                        st.session_state.step_status["retrieve"] = "done"
                        st.session_state.step_status["draft"] = "done"
                        st.session_state.step_status["critique"] = "done"

                        if res.status_code == 200:
                            data = res.json()
                            if data.get("status") == "completed":
                                st.session_state.step_status["finalize"] = "done"
                                st.session_state.final_answer = data["final_answer"]
                                st.session_state.history.append({
                                    "query": active_query,
                                    "answer": data["final_answer"],
                                    "status": "Auto-Finalized",
                                })
                            elif data.get("status") == "pending_review":
                                st.session_state.step_status["human_review"] = "running"
                                st.session_state.pending_thread_id = data["thread_id"]
                                st.session_state.pending_draft = data["draft_answer"]
                                st.session_state.pending_critique = data["critique"]
                        else:
                            st.error(f"Pipeline Execution Failed: {res.text}")
                    except Exception as err:
                        st.error(f"Connection Error: {err}")

    # ---- 3. Reviewer Sign-Off Desk (Human-in-the-Loop) ----
    if st.session_state.pending_thread_id:
        st.markdown("""
        <div class="reviewer-banner">
            <div class="eyebrow" style="color: #F59E0B;">EXPERT REVIEWER SIGN-OFF REQUIRED</div>
            <div style="font-size: 16px; font-weight: 700; color: #FBBF24; margin-bottom: 8px;">
                ⚠️ AI Draft Flagged for Human Judgment or Revision
            </div>
            <div style="font-size: 13px; color: #CBD5E1;">
                This query involves risk evaluation or triggering critique checks. Please review the AI draft below and either approve it as-is or provide a corrected legal assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### AI Draft Answer")
        st.info(st.session_state.pending_draft)

        with st.expander("🔍 View AI Fact-Checker Critique & Reasoning Notes"):
            st.markdown(st.session_state.pending_critique)

        correction = st.text_area(
            "Reviewer Assessment",
            placeholder="Leave blank to approve draft as-is, or enter your modified/corrected legal assessment here...",
            label_visibility="collapsed",
            height=100,
        )

        if st.button("✅ Sign Off & Finalize Audit"):
            with st.spinner("Finalizing audit decision..."):
                payload = {
                    "thread_id": st.session_state.pending_thread_id,
                    "correction": correction.strip() if correction.strip() else None,
                }
                try:
                    res = requests.post(f"{API_BASE}/review/resume", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.step_status["human_review"] = "done"
                        st.session_state.step_status["finalize"] = "done"
                        st.session_state.final_answer = data["final_answer"]
                        st.session_state.history.append({
                            "query": st.session_state.query_input,
                            "answer": data["final_answer"],
                            "status": "Human Approved" if not correction.strip() else "Human Modified",
                        })
                        st.session_state.pending_thread_id = None
                        st.rerun()
                    else:
                        st.error(f"Resume action failed: {res.text}")
                except Exception as ex:
                    st.error(f"Error resuming pipeline: {ex}")

    # ---- 4. Final Answer Output Card ----
    if st.session_state.final_answer and not st.session_state.pending_thread_id:
        st.markdown("""
        <div class="answer-card">
            <div class="eyebrow">FINAL AUDIT REPORT</div>
            <div style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;">Verified Legal Answer</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.final_answer)

with right_col:
    # ---- Pipeline Status Panel ----
    st.markdown('<div class="eyebrow">PIPELINE EXECUTION MONITOR</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 16px;">Graph Agent Workflow</div>', unsafe_allow_html=True)

    for i, (key, title, desc) in enumerate(PIPELINE_STEPS, start=1):
        status = st.session_state.step_status[key]
        badge_label = {"waiting": "WAITING", "running": "RUNNING...", "done": "DONE ✓"}[status]
        st.markdown(f"""
        <div class="pipeline-card {status}">
            <span class="status-badge {status}">{badge_label}</span>
            <span class="step-num">STEP {i:02d}</span>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Audit History Panel ----
    if st.session_state.history:
        st.write("")
        st.markdown('<div class="eyebrow">RECENT AUDITS IN THIS SESSION</div>', unsafe_allow_html=True)
        for idx, item in enumerate(reversed(st.session_state.history), start=1):
            with st.expander(f"📌 {item['query'][:40]}... ({item['status']})"):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Final Answer:**\n{item['answer']}")