"""
Streamlit frontend for LegalLens — calls the FastAPI backend over HTTP.
Run the backend first: uvicorn main:app --reload
Then run this: streamlit run streamlit_app.py
"""
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="LegalLens", page_icon="⚖️", layout="wide")

# ---- Custom CSS: dark theme, teal accent ----
st.markdown("""
<style>
    .stApp {
        background-color: #0a0f0d;
        color: #e8e6e0;
    }
    .eyebrow {
        color: #5DCAA5;
        font-size: 13px;
        letter-spacing: 3px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 72px;
        font-weight: 800;
        background: linear-gradient(90deg, #5DCAA5, #1D9E75);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 12px;
    }
    .hero-sub {
        color: #9c9a92;
        font-size: 18px;
        max-width: 640px;
    }
    .pipeline-card {
        background-color: #121815;
        border: 1px solid #1e2622;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 14px;
    }
    .pipeline-card.done {
        border-color: #1D9E75;
    }
    .pipeline-card.running {
        border-color: #5DCAA5;
        box-shadow: 0 0 12px rgba(93, 202, 165, 0.25);
    }
    .step-num {
        color: #5DCAA5;
        font-weight: 700;
        font-size: 13px;
    }
    .step-title {
        color: #e8e6e0;
        font-weight: 700;
        font-size: 17px;
        display: inline;
        margin-left: 8px;
    }
    .step-desc {
        color: #7a786f;
        font-size: 13px;
        margin-top: 6px;
    }
    .status-badge {
        float: right;
        font-size: 11px;
        letter-spacing: 1px;
        color: #7a786f;
    }
    .status-badge.done { color: #5DCAA5; }
    .status-badge.running { color: #5DCAA5; }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background-color: #121815 !important;
        color: #e8e6e0 !important;
        border: 1px solid #1e2622 !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #5DCAA5, #1D9E75);
        color: #04342C;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 0;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ---- Session state ----
for key, default in {
    "document_id": None, "document_path": None,
    "pending_thread_id": None, "pending_draft": None, "pending_critique": None,
    "step_status": {"retrieve": "waiting", "draft": "waiting", "critique": "waiting",
                     "human_review": "waiting", "finalize": "waiting"},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

PIPELINE_STEPS = [
    ("retrieve", "Retrieve", "Pulls relevant clauses via hybrid search"),
    ("draft", "Draft", "Generates a grounded first-pass answer"),
    ("critique", "Critique", "Fact-checks the draft against source clauses"),
    ("human_review", "Human review", "Expert sign-off on judgment calls"),
    ("finalize", "Finalize", "Returns the confirmed answer"),
]

def reset_pipeline_status():
    st.session_state.step_status = {k: "waiting" for k, _, _ in PIPELINE_STEPS}

def render_pipeline():
    st.markdown("### Pipeline")
    for key, title, desc in PIPELINE_STEPS:
        status = st.session_state.step_status[key]
        badge = {"waiting": "WAITING", "running": "RUNNING", "done": "DONE"}[status]
        st.markdown(f"""
        <div class="pipeline-card {status}">
            <span class="step-num">{PIPELINE_STEPS.index((key, title, desc))+1:02d}</span>
            <span class="step-title">{title}</span>
            <span class="status-badge {status}">{badge}</span>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ---- Hero ----
st.markdown('<div class="eyebrow">LEGAL REVIEW ASSISTANT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">LegalLens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A RAG pipeline with self-correction and human sign-off — '
    'built for legal professionals reviewing contracts, not a replacement for legal advice.</div>',
    unsafe_allow_html=True
)
st.write("")
st.write("")

left, right = st.columns([1.3, 1])

with left:
    st.markdown('<div class="eyebrow">DOCUMENT</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF or TXT contract", type=["pdf", "txt"], label_visibility="collapsed")

    if uploaded_file is not None and st.button("Index document"):
        with st.spinner("Indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_BASE}/upload", files=files)
        if response.status_code == 200:
            data = response.json()
            st.session_state.document_id = data["document_id"]
            st.session_state.document_path = data["document_path"]
            st.success(f"Indexed — document_id: {data['document_id']}")
        else:
            st.error(f"Upload failed: {response.text}")

    if st.session_state.document_id:
        st.markdown('<div class="eyebrow" style="margin-top:24px">QUESTION</div>', unsafe_allow_html=True)
        query = st.text_input("query", placeholder="e.g. Is the security deposit clause risky for the tenant?", label_visibility="collapsed")

        st.caption("Try →")
        cols = st.columns(3)
        examples = ["What is the monthly rent?", "Is this deposit clause risky?", "What is the notice period?"]
        for c, ex in zip(cols, examples):
            if c.button(ex, key=ex):
                query = ex

        if st.button("⚡ Run review pipeline") and query:
            reset_pipeline_status()
            st.session_state.step_status["retrieve"] = "running"

            with st.spinner("Analyzing..."):
                payload = {
                    "document_id": st.session_state.document_id,
                    "document_path": st.session_state.document_path,
                    "query": query,
                }
                response = requests.post(f"{API_BASE}/review", json=payload)

            st.session_state.step_status["retrieve"] = "done"
            st.session_state.step_status["draft"] = "done"
            st.session_state.step_status["critique"] = "done"

            if response.status_code == 200:
                data = response.json()
                if data["status"] == "completed":
                    st.session_state.step_status["finalize"] = "done"
                    st.success("Answer")
                    st.write(data["final_answer"])
                elif data["status"] == "pending_review":
                    st.session_state.step_status["human_review"] = "running"
                    st.session_state.pending_thread_id = data["thread_id"]
                    st.session_state.pending_draft = data["draft_answer"]
                    st.session_state.pending_critique = data["critique"]
                    st.warning("Awaiting reviewer sign-off below.")
            else:
                st.error(f"Request failed: {response.text}")

    if st.session_state.pending_thread_id:
        st.markdown('<div class="eyebrow" style="margin-top:24px">REVIEWER SIGN-OFF</div>', unsafe_allow_html=True)
        st.info(st.session_state.pending_draft)
        with st.expander("Critique / self-check notes"):
            st.write(st.session_state.pending_critique)
        correction = st.text_area("Leave blank to approve, or enter your corrected assessment", label_visibility="collapsed")

        if st.button("Submit decision"):
            with st.spinner("Finalizing..."):
                payload = {"thread_id": st.session_state.pending_thread_id, "correction": correction or None}
                response = requests.post(f"{API_BASE}/review/resume", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.session_state.step_status["human_review"] = "done"
                st.session_state.step_status["finalize"] = "done"
                st.success("Finalized")
                st.write(data["final_answer"])
                st.session_state.pending_thread_id = None
            else:
                st.error(f"Resume failed: {response.text}")

with right:
    render_pipeline()