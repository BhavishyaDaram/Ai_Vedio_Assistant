import os
import html
import json
import streamlit as st
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# True on Streamlit Cloud, False on your laptop
IS_CLOUD = os.path.exists("/mount/src") or os.getenv("HOME") == "/home/adminuser"

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Ember & Teal theme) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0f0a08;
    --surface: #1a120d;
    --surface-2: #241a13;
    --border: #3a2a1f;
    --accent: #ff7849;
    --accent-glow: #ffab73;
    --accent-2: #2dd4bf;
    --text: #f5ede4;
    --text-muted: #a08b76;
    --success: #10b981;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(255,120,73,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,120,73,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* Hero */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800; line-height: 1.1; margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 0.8rem; color: var(--text-muted) !important;
    letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.5rem;
}

/* Cards (bordered containers) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--accent) !important; }

.card-title {
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 0.5rem;
}
.session-title {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: var(--text);
}

/* Badges */
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.65rem;
         font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; }
.badge-orange { background: rgba(255,120,73,0.18); color: var(--accent-glow); border: 1px solid rgba(255,120,73,0.3); }
.badge-cyan   { background: rgba(45,212,191,0.15); color: var(--accent-2); border: 1px solid rgba(45,212,191,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(255,120,73,0.2) !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), #c2410c) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.875rem !important; letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important; transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(255,120,73,0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'Syne', sans-serif; font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* Chat */
[data-testid="stChatInput"] { border: 1px solid var(--accent) !important; border-radius: 10px !important; background: var(--surface-2) !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stChatMessage"] {
    background: var(--surface) !important; border: 1px solid var(--border);
    border-radius: 12px; padding: 0.75rem 1rem;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.transcript-box {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.25rem; font-size: 0.82rem; line-height: 1.8; max-height: 500px;
    overflow-y: auto; color: var(--text-muted); white-space: pre-wrap; word-break: break-word;
}

/* Misc */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
for key, default in {"result": None, "chat_history": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helpers ────────────────────────────────────────────────────────────────────
def reset_session():
    st.session_state.result = None
    st.session_state.chat_history = []


def card(title: str, content: str):
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(content)


def build_report_md(r: dict) -> str:
    return (
        f"# {r['title']}\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action Items\n{r['action_items']}\n\n"
        f"## Key Decisions\n{r['key_decisions']}\n\n"
        f"## Open Questions\n{r['open_questions']}\n"
    )


def save_upload(uploaded) -> str:
    path = os.path.join(DOWNLOAD_DIR, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    input_mode = st.radio(
        "Input Source",
        ["YouTube URL", "Upload File"],
        index=1 if IS_CLOUD else 0,  # default to Upload File when hosted
    )

    if IS_CLOUD and input_mode == "YouTube URL":
        st.warning(
            "YouTube often blocks downloads from hosted servers, so this may "
            "fail with a 403 error. For best results, choose **Upload File** "
            "or run the app locally."
        )

    source = ""
    uploaded_file = None
    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube URL / File Path",
            placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload audio / video",
            type=["mp4", "mkv", "mov", "webm", "mp3", "wav", "m4a"],
        )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.result:
        st.markdown("---")
        if st.button("🔄  New Analysis", use_container_width=True):
            reset_session()
            st.rerun()

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Run Pipeline (live progress via st.status) ─────────────────────────────────
if run_btn:
    if input_mode == "Upload File" and uploaded_file is None:
        st.error("Please upload a file first.")
    elif input_mode == "YouTube URL" and not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        reset_session()
        try:
            with st.status("⚙️ Starting pipeline…", expanded=True) as status_box:
                if input_mode == "Upload File":
                    status_box.write("📁 Saving uploaded file…")
                    src = save_upload(uploaded_file)
                else:
                    src = source.strip()

                status_box.write("🔊 Processing input (downloading / chunking audio)…")
                chunks = process_input(src)

                status_box.write("📝 Transcribing audio… (this can take a while for long videos)")
                transcript = transcribe_all(chunks, language)
                status_box.write(f"↳ Got {len(transcript)} characters of transcript.")

                status_box.write("🏷️ Generating title…")
                title = generate_title(transcript)

                status_box.write("📋 Summarising…")
                summary = summarize(transcript)

                status_box.write("🔍 Extracting action items, decisions & questions…")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                status_box.write("🧠 Building RAG chat engine…")
                rag_chain = build_rag_chain(transcript)

                status_box.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.rerun()
        except Exception as e:
            msg = str(e)
            if "403" in msg or "Forbidden" in msg:
                st.error(
                    "❌ YouTube blocked this download (HTTP 403). This happens on "
                    "hosted servers. Please use **Upload File** instead, or run the app locally."
                )
            else:
                st.error(f"❌ Error: {msg}")

# ─── Results ────────────────────────────────────────────────────────────────────
r = st.session_state.result

if r:
    # Session title
    with st.container(border=True):
        st.markdown('<div class="card-title">📌 Session Title</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="session-title">{html.escape(str(r["title"]))}</div>', unsafe_allow_html=True)

    tab_summary, tab_chat, tab_transcript, tab_export = st.tabs(
        ["📋 Summary & Takeaways", "💬 Chat with Meeting (RAG)", "📄 Full Transcript", "📤 Export Reports"]
    )

    # ── Summary & Takeaways ──
    with tab_summary:
        card("📋 Summary", r["summary"])
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            card("✅ Action Items", r["action_items"])
        with c2:
            card("🔑 Key Decisions", r["key_decisions"])
        with c3:
            card("❓ Open Questions", r["open_questions"])

    # ── Chat ──
    with tab_chat:
        st.markdown("### 💬 Interactive Meeting Assistant")
        st.caption("Ask questions about decisions, action items, or specific topics from this meeting.")

        history_box = st.container()
        question = st.chat_input("Ask a question about your meeting...")

        with history_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if question and question.strip():
                q = question.strip()
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = ask_question(r["rag_chain"], q)
                        except Exception as e:
                            answer = f"⚠️ Error answering question: {e}"
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ── Transcript ──
    with tab_transcript:
        st.markdown(
            f'<div class="transcript-box">{html.escape(r["transcript"])}</div>',
            unsafe_allow_html=True,
        )

    # ── Export ──
    with tab_export:
        st.markdown("### 📤 Export Reports")
        st.caption("Download the analysis in the format you need.")
        e1, e2, e3 = st.columns(3, gap="medium")
        with e1:
            st.download_button(
                "⬇️ Full report (.md)",
                data=build_report_md(r),
                file_name="meeting_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with e2:
            st.download_button(
                "⬇️ Transcript (.txt)",
                data=r["transcript"],
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with e3:
            st.download_button(
                "⬇️ Data (.json)",
                data=json.dumps(
                    {k: v for k, v in r.items() if k != "rag_chain"},
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                file_name="meeting_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Upload a file (or paste a YouTube URL when running locally) in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-orange">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)
