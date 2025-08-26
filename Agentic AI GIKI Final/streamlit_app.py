# streamlit_app.py
import os, time, uuid, re
import streamlit as st
from dotenv import load_dotenv
from src.agents.controller import build_controller

# ---------- Simple detectors / helpers ----------
_PURE_EXPR_RE = re.compile(r"^[0-9\.\s\+\-\*\/\%\(\)]+(?:\*\*)?$")

def _looks_like_pure_expr(s: str) -> bool:
    return bool(_PURE_EXPR_RE.fullmatch(s or ""))

def _build_context(history, max_turns=6, max_chars=1200) -> str:
    recent = history[-max_turns:]
    lines = []
    for t in recent:
        q = (t.get("q","") or "").strip().replace("\n", " ")
        a = (t.get("a","") or "").strip().replace("\n", " ")
        if q and a:
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
    ctx = "\n".join(lines)
    return (ctx[:max_chars] + "…") if len(ctx) > max_chars else ctx

def _as_text(resp) -> str:
    txt = getattr(resp, "content", None)
    if isinstance(txt, str):
        return txt.strip()
    return str(resp).strip()

def _clean(ans: str) -> str:
    return ans.replace("$$", "").replace("$", "").replace("\\boxed{", "").replace("}", "").strip()

def _tool_calls(resp):
    try:
        if hasattr(resp, "formatted_tool_calls") and resp.formatted_tool_calls:
            return ", ".join(resp.formatted_tool_calls)
        if hasattr(resp, "tools") and resp.tools:
            return ", ".join([t.tool_name for t in resp.tools if getattr(t, "tool_name", None)])
    except Exception:
        pass
    return None

def _save_uploads(files, folder: str) -> list[str]:
    os.makedirs(folder, exist_ok=True)
    saved = []
    for f in files or []:
        path = os.path.join(folder, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved.append(path)
    return saved

# ---------- UI ----------
st.set_page_config(page_title="Agent UI (OpenAI)", page_icon="🤖", layout="centered")
st.title("🤖 Tool-Calling Agent")
st.caption("Controller + Tools (Web Search, Calculator, GSM8k Solver, RAG) • OpenAI API")

# ---------- SIDEBAR: minimal ----------
with st.sidebar:
    st.subheader("Settings")
    load_dotenv()
    rag_folder = st.text_input("RAG folder (for files)", value=os.getenv("RAG_FOLDER", "data"))
    # NEW: keep env in sync with the textbox
    os.environ["RAG_FOLDER"] = rag_folder
    temperature = st.slider("Controller temperature", 0.0, 1.0, 0.2, 0.1)
    init = st.button("Initialize / Rebuild Agent")
    clear = st.button("Clear Chat")

# ---------- INIT ----------
if "agent" not in st.session_state or init:
    model = os.getenv("MODEL", "gpt-4o-mini")  # read from env only
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set OPENAI_API_KEY in your environment or .env file.")
    st.session_state.agent = build_controller(model=model, temperature=temperature)
    st.session_state.history = []
    st.session_state.session_id = str(uuid.uuid4())
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    st.sidebar.success(f"Agent ready (model={model})")

if clear:
    st.session_state.history = []
    st.session_state.uploaded_files = []
    st.session_state.session_id = str(uuid.uuid4())
    st.success("Chat cleared.")

# ---------- CHAT HISTORY ----------
chat_container = st.container()
with chat_container:
    for turn in st.session_state.get("history", []):
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["a"])
            if turn.get("tool"):
                st.caption(f"🛠 Tool: {turn['tool']}")
            st.caption(f"⏱ {turn['latency']:.2f}s")

# ---------- ATTACHMENT ROW ----------
st.markdown("#### Attach files (PDF / DOCX / TXT)")
up_col1, up_col2 = st.columns([3, 2])
with up_col1:
    uploads = st.file_uploader(
        "Drop files here (saved to your RAG folder)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with up_col2:
    st.caption(f"Save location: `./{rag_folder}`")


if uploads:
    saved_paths = _save_uploads(uploads, rag_folder)
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    st.session_state.uploaded_files.extend([os.path.basename(p) for p in saved_paths])
    st.success(f"Saved {len(saved_paths)} file(s).")

if st.session_state.get("uploaded_files"):
    st.caption("Attached this session:")
    st.write(" • " + " • ".join(st.session_state.uploaded_files))

st.markdown("---")

# ---------- BOTTOM CHAT INPUT ----------
user_text = st.chat_input("Type your message… (e.g., search the local docs: fee schedule | or: solve word problems from docs)")

if user_text:
    with st.chat_message("user"):
        st.write(user_text)

    msg = user_text.strip()
    if st.session_state.history and not _looks_like_pure_expr(msg):
        ctx = _build_context(st.session_state.history, max_turns=6, max_chars=1200)
        if ctx:
            msg = f"[Context]\n{ctx}\n\n[Request]\n{msg}"

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            resp = st.session_state.agent.run(
                msg,
                session_id=st.session_state.session_id,
                user_id="streamlit",
            )
            dt = time.time() - t0
            ans = _clean(_as_text(resp))
            tool = _tool_calls(resp)

        st.write(ans or "(empty)")
        if tool:
            st.caption(f"🛠 Tool: {tool}")
        st.caption(f"⏱ {dt:.2f}s")

    st.session_state.history.append({"q": user_text, "a": ans, "tool": tool, "latency": dt})
