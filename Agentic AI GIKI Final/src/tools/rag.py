# src/tools/rag.py

import os, glob, re, hashlib
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from chromadb.errors import InvalidArgumentError
from src.tools.gsm8k_solver import gsm8k_solve

# Optional DOCX support
try:
    from docx import Document  # python-docx
except Exception:
    Document = None  # we'll skip .docx if not installed

COLLECTION_NAME = "local_kb1"  # keeping your original collection name
_DB = None

_EMBED = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ---------- DB ----------
def _get_db():
    global _DB
    if _DB is None:
        _DB = chromadb.Client()  # switch to PersistentClient(...) if you want persistence
    return _DB

# ---------- Readers ----------
def _read_pdf(path: str) -> str:
    parts = []
    with open(path, "rb") as f:
        r = PdfReader(f)
        for p in r.pages:
            parts.append(p.extract_text() or "")
    return "\n".join(parts)

def _read_txt(path: str) -> str:
    # robust text decode
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    # last resort: binary read then decode ignore
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _read_docx(path: str) -> str:
    if Document is None:
        return ""  # python-docx not installed
    try:
        doc = Document(path)
        return "\n".join(p.text or "" for p in doc.paragraphs)
    except Exception:
        return ""

def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".txt":
        return _read_txt(path)
    if ext == ".docx":
        return _read_docx(path)
    # NOTE: legacy .doc (binary) not supported without extra deps (e.g., textract/mammoth)
    return ""

# ---------- Files ----------
def _list_supported_files(folder: str) -> list[str]:
    """Find PDF, DOCX, and TXT files (case-insensitive)."""
    globs = [
        "*.pdf", "*.PDF",
        "*.docx", "*.DOCX",
        "*.txt", "*.TXT",
    ]
    files: list[str] = []
    for pat in globs:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files))

# ---------- Chroma helpers ----------
def _safe_collection(db, name: str):
    """Create or get a Chroma collection with a valid name."""
    valid = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("._-") or "local_kb1"
    try:
        return db.get_or_create_collection(valid, embedding_function=_EMBED)
    except InvalidArgumentError:
        return db.get_or_create_collection("local_kb1", embedding_function=_EMBED)

# ---------- RAG: query ----------
def rag_query(question: str, folder: str | None = None) -> str:
    folder = (folder or os.getenv("RAG_FOLDER", "data")).strip()
    folder_abs = os.path.abspath(folder)
    db = _get_db()
    coll = _safe_collection(db, COLLECTION_NAME)

    # Build index once
    try:
        needs_load = (coll.count() == 0)
    except Exception:
        needs_load = True

    if needs_load:
        files = _list_supported_files(folder)
        if not files:
            return (
                "No supported files found in the folder.\n"
                f"Scanned: {folder_abs}\n"
                "Tip: upload PDF/DOCX/TXT via sidebar or put them in ./data"
            )
        docs: List[str] = []
        ids: List[str] = []
        for i, path in enumerate(files):
            try:
                text = _read_file(path)
                if text.strip():
                    docs.append(text)
                    # simple, stable-ish id using filename + mtime
                    fid = f"{os.path.basename(path)}-{int(os.path.getmtime(path))}-{i}"
                    ids.append(fid)
            except Exception:
                pass
        if not docs:
            return f"No readable PDF/DOCX/TXT found in: {folder_abs}"
        coll.add(documents=docs, ids=ids)

    hits = coll.query(query_texts=[question], n_results=4)
    chunks = hits.get("documents", [[]])[0]
    if not chunks:
        return "No matches in the local KB."
    context = "\n\n".join(chunks)
    return f"Top matches from local KB ({folder_abs}):\n{context[:1500]}"

# ---------- Word-problem extraction ----------
def _extract_word_problems(text: str, max_q: int = 10) -> List[str]:
    keywords = r"(how many|how much|left|remaining|total|altogether|per|rate|cost|price|profit|loss|each|share|split|average|percent|percentage|ratio|discount|speed|time|distance|groups|seats|students|apples|rupees|km|minutes|hours)"
    has_num = r"\d"
    cand = re.split(r"(?<=[\.\?\!])\s+", text)
    out: List[str] = []
    for s in cand:
        s_clean = s.strip()
        if not s_clean:
            continue
        if re.search(has_num, s_clean, re.I) and re.search(keywords, s_clean, re.I):
            out.append(s_clean)
        elif s_clean.endswith("?") and re.search(has_num, s_clean, re.I):
            out.append(s_clean)
        if len(out) >= max_q:
            break
    seen, unique = set(), []
    for q in out:
        if q not in seen:
            seen.add(q); unique.append(q)
    return unique

# ---------- RAG: solve math from docs ----------
def rag_solve_math_from_docs(prompt: str = "", folder: str | None = None, max_questions: int = 10) -> str:
    folder = (folder or os.getenv("RAG_FOLDER", "data")).strip()
    folder_abs = os.path.abspath(folder)
    db = _get_db()
    coll = _safe_collection(db, COLLECTION_NAME)

    # Build index once (same as rag_query)
    try:
        needs_load = (coll.count() == 0)
    except Exception:
        needs_load = True

    if needs_load:
        files = _list_supported_files(folder)
        if not files:
            return (
                "No supported files found in the folder.\n"
                f"Scanned: {folder_abs}\n"
                "Tip: upload PDF/DOCX/TXT via sidebar or put them in ./data"
            )
        docs: List[str] = []
        ids: List[str] = []
        for i, path in enumerate(files):
            try:
                text = _read_file(path)
                if text.strip():
                    docs.append(text)
                    fid = f"{os.path.basename(path)}-{int(os.path.getmtime(path))}-{i}"
                    ids.append(fid)
            except Exception:
                pass
        if not docs:
            return f"No readable PDF/DOCX/TXT found in: {folder_abs}"
        coll.add(documents=docs, ids=ids)

    hits = coll.query(query_texts=[prompt or "math word problems"], n_results=6)
    docs_lists = hits.get("documents", [[]])
    if not docs_lists or not docs_lists[0]:
        return "No matches in the local KB."

    candidate_text = "\n\n".join(docs_lists[0])
    questions = _extract_word_problems(candidate_text, max_q=max_questions)
    if not questions:
        return f"No math word problems were detected in documents under: {folder_abs}"

    lines: List[str] = []
    for i, q in enumerate(questions, start=1):
        sol = gsm8k_solve(q)  # Reasoning + Answer
        lines.append(f"{i}. Question: {q}\n{sol}\n")

    return f"Solved math word problems from local docs ({folder_abs}):\n\n" + "\n".join(lines)
