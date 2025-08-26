# src/eval/lama_agent_eval.py
# LAMA agent benchmark (controller w/ tools)
# - Small, diverse pool loader (row-group streaming + shuffle + dedup)
# - Strict [MASK] cloze prompt
# - Agent watchdog timeout
# - Metrics: pred_acc, tool_acc (web_search used), joint_acc
#
# Usage:
#   pip install -U huggingface_hub pandas pyarrow fsspec python-dotenv
#   export GROQ_API_KEY=YOUR_KEY; export OPENAI_API_KEY=$GROQ_API_KEY
#   python -m src.eval.lama_agent_eval \
#     --configs trex google_re conceptnet squad \
#     --n 20 \
#     --model openai/gpt-oss-20b \
#     --csv logs/lama_agent_run.csv
#
# Env knobs (optional):
#   LAMA_TARGET_ROWS=200         # pool size to read (default 200)
#   AGENT_TIMEOUT_S=25           # per-item timeout
#   WEB_SEARCH_PACE_S=0.15       # delay between items (be kind to search)

import os, csv, time, json, re, argparse, random
from typing import List, Dict, Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO

from src.agents.controller import build_controller

# -------------------------
# Configs & constants
# -------------------------
VALID_CONFIGS = {"trex", "google_re", "conceptnet", "squad"}
CANDIDATE_COLS = [
    "masked_sentence",
    "obj_label", "obj_surface", "obj", "object", "label",
    "obj_aliases", "aliases", "object_aliases",
]

AGENT_TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", "25"))
PACE_S = float(os.getenv("WEB_SEARCH_PACE_S", "0.0"))  # small pacing per item (seconds)

# -------------------------
# Env helpers
# -------------------------
def _ensure_keys_loaded():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and os.getenv("GROQ_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------
# Repo fetch (Parquet ref)
# -------------------------
def _repo_local_path() -> str:
    """
    Download facebook/lama Parquet shards to a local cache.
    Parquet data is on the 'refs/convert/parquet' ref.
    """
    cache_dir = os.path.join(os.getcwd(), ".hf_cache_lama")
    os.makedirs(cache_dir, exist_ok=True)
    local_dir = snapshot_download(
        repo_id="facebook/lama",
        repo_type="dataset",
        revision="refs/convert/parquet",
        local_dir=cache_dir,
        allow_patterns=[
            "trex/train/*.parquet",
            "google_re/train/*.parquet",
            "conceptnet/train/*.parquet",
            "squad/train/*.parquet",
        ],
    )
    return local_dir

def _list_parquet_files(base_dir: str, config: str) -> list[str]:
    root = os.path.join(base_dir, config, "train")
    files = []
    for r, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.lower().endswith(".parquet"):
                files.append(os.path.join(r, fn))
    if not files:
        alt = os.path.join(base_dir, config)
        for r, _, fnames in os.walk(alt):
            for fn in fnames:
                if fn.lower().endswith(".parquet"):
                    files.append(os.path.join(r, fn))
    return sorted(files)

# -------------------------
# Pool loader (diverse + early stop)
# -------------------------
def _read_needed_rows(files: list[str], target_rows: int = 200, verbose: bool = True) -> pd.DataFrame:
    """
    Stream row-groups from each parquet and keep only rows containing [MASK].
    Randomize file order and sample per row-group for diversity.
    Stop as soon as ~target_rows are collected (then dedup masked_sentence).
    """
    rows: list[pd.DataFrame] = []
    needed_cols: list[str] | None = None
    collected = 0

    # Allow override via env
    try:
        env_rows = int(os.getenv("LAMA_TARGET_ROWS", "0"))
        if env_rows > 0:
            target_rows = env_rows
    except Exception:
        pass

    files = list(files)
    random.shuffle(files)  # promote cross-file diversity

    total_files = len(files)
    for idx, path in enumerate(files, 1):
        try:
            schema = pq.read_schema(path)
            cols_here = [c for c in CANDIDATE_COLS if c in schema.names]
            if "masked_sentence" not in cols_here:
                if verbose:
                    print(f"[LAMA] {idx}/{total_files} skip (no masked_sentence): {os.path.basename(path)}", flush=True)
                continue
            if needed_cols is None:
                needed_cols = cols_here

            pf = pq.ParquetFile(path)
            kept_this_file = 0
            for rg in range(pf.num_row_groups):
                need = max(0, target_rows - collected)
                if need == 0:
                    break

                tbl = pf.read_row_group(rg, columns=needed_cols)
                dfg = tbl.to_pandas()
                dfg = dfg[dfg["masked_sentence"].astype(str).str.contains(r"\[MASK\]", na=False)]

                if not len(dfg):
                    continue

                # sample per group for diversity (bounded by 'need')
                take = min(need, len(dfg))
                seed = (hash(path) ^ (rg + 1) ^ 0xA5A5_1234) & 0xFFFFFFFF
                dfg = dfg.sample(n=take, random_state=seed)

                rows.append(dfg)
                collected += len(dfg)
                kept_this_file += len(dfg)

                if verbose:
                    print(f"[LAMA] {idx}/{total_files} rg={rg+1}/{pf.num_row_groups} — kept {len(dfg)} (total {collected}/{target_rows})", flush=True)

                if collected >= target_rows:
                    break

            if verbose and kept_this_file == 0:
                print(f"[LAMA] {idx}/{total_files} read — kept 0 (total {collected}/{target_rows})", flush=True)

            if collected >= target_rows:
                break

        except Exception as e:
            if verbose:
                print(f"[LAMA] {idx}/{total_files} error — {os.path.basename(path)}: {e}", flush=True)
            continue

    if not rows:
        return pd.DataFrame(columns=["masked_sentence"])

    df = pd.concat(rows, ignore_index=True)

    # Deduplicate identical sentences to avoid clones
    if "masked_sentence" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["masked_sentence"]).reset_index(drop=True)
        if verbose and len(df) != before:
            print(f"[LAMA] dedup masked_sentence: {before} → {len(df)}", flush=True)

    return df

def load_lama_df(config: str, sample_cap: int | None = None, verbose: bool = True) -> pd.DataFrame:
    assert config in VALID_CONFIGS, f"Unknown LAMA config: {config}"
    base = _repo_local_path()
    files = _list_parquet_files(base, config)
    if not files:
        raise RuntimeError(f"No parquet files found for LAMA/{config}")

    # Pool size: default 200 rows total (diverse). Can override with LAMA_TARGET_ROWS.
    pool_target = int(os.getenv("LAMA_TARGET_ROWS", "200"))
    df = _read_needed_rows(files, target_rows=pool_target, verbose=verbose)

    if "masked_sentence" not in df.columns:
        raise RuntimeError(f"'masked_sentence' missing in LAMA/{config}; columns: {list(df.columns)[:20]}")
    df = df.dropna(subset=["masked_sentence"]).reset_index(drop=True)
    return df

# -------------------------
# Scoring helpers
# -------------------------
def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r'^[\"“”\'\(\)\[\]\s]+|[\"“”\'\(\)\[\]\s]+$', "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _json_or_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

def gold_variants_row(row: pd.Series) -> List[str]:
    """Collect acceptable gold strings across schema variants, including aliases."""
    cands: list[str] = []
    for key in ["obj_label", "obj_surface", "obj", "object", "label"]:
        if key in row and pd.notna(row[key]):
            cands.append(str(row[key]))
    for k in ["obj_aliases", "aliases", "object_aliases"]:
        if k in row and pd.notna(row[k]):
            for a in _json_or_list(row[k]):
                if a:
                    cands.append(str(a))
    out, seen = [], set()
    for c in cands:
        n = normalize(c)
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out or [""]

# -------------------------
# Agent helpers
# -------------------------
def _as_text(resp) -> str:
    txt = getattr(resp, "content", None)
    if isinstance(txt, str):
        return txt.strip()
    return str(resp).strip()

def _tool_calls(resp) -> str | None:
    try:
        if hasattr(resp, "formatted_tool_calls") and resp.formatted_tool_calls:
            return ", ".join(resp.formatted_tool_calls)
        if hasattr(resp, "tools") and resp.tools:
            names = [getattr(t, "tool_name", None) or getattr(t, "tool", None) for t in resp.tools]
            names = [n for n in names if n]
            if names:
                return ", ".join(names)
    except Exception:
        pass
    return None

def _run_with_timeout(agent, prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(agent.run, prompt)
        return fut.result(timeout=AGENT_TIMEOUT_S)

def one_token(s: str) -> str:
    """Clamp to a single alpha-ish token for cloze answers."""
    m = re.search(r"[A-Za-z][A-Za-z\-']*", s or "")
    return m.group(0) if m else (s or "").strip().split()[0] if s else ""

def build_cloze_prompt(masked_sentence: str) -> str:
    """
    Strong guidance to prefer web_search for factual cloze and to emit one token.
    """
    return (
        "Task: Fill the [MASK] in the sentence with the correct single word.\n"
        "Routing hint: Prefer web_search for factual entities; do not guess.\n"
        "Output rule: Return ONLY the missing word (no quotes, no punctuation, no extra text).\n"
        "Special cases:\n"
        " - If the pattern is 'CITY, [MASK]', choose the country/region after the comma.\n"
        " - If the pattern is 'the state capital of [MASK]' and a city is to the left, output the jurisdiction (e.g., 'Victoria').\n\n"
        f"Sentence: {masked_sentence}\n"
    )

# -------------------------
# Evaluate per config (agent-based)
# -------------------------
def eval_config_with_agent(agent, config: str, n: int, writer: csv.writer, run_id: str, verbose: bool = True) -> Dict[str, Any]:
    df = load_lama_df(config, sample_cap=n, verbose=verbose)
    df = df[df["masked_sentence"].astype(str).str.contains(r"\[MASK\]", na=False)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"No masked sentences with [MASK] found in LAMA/{config}")

    # sample to exactly n (from a diverse, deduped pool)
    if n:
        take = min(n, len(df))
        df_eval = df.sample(n=take, random_state=0).reset_index(drop=True)
    else:
        df_eval = df

    correct = 0
    total = len(df_eval)
    latencies: List[float] = []
    tool_hits = 0
    joint_hits = 0

    for i, row in enumerate(df_eval.iterrows(), start=1):
        _, r = row
        masked = str(r["masked_sentence"])
        golds = gold_variants_row(r)
        prompt = build_cloze_prompt(masked)

        t0 = time.time()
        try:
            resp = _run_with_timeout(agent, prompt)
            raw = _as_text(resp)
            tool = _tool_calls(resp)
        except _TO:
            raw = "timeout"
            tool = "timeout"
        dt = time.time() - t0

        # Keep only first line; strip quotes/punct; clamp to one token
        pred = raw.split("\n", 1)[0].strip().strip("\"'“”").rstrip(".,;:)")
        pred = one_token(pred)

        ok = normalize(pred) in golds

        # tool correctness: did we use web_search at least once?
        tool_str = tool or ""
        used_web = "web_search" in tool_str

        latencies.append(dt)
        correct += 1 if ok else 0
        tool_hits += 1 if used_web else 0
        joint_hits += 1 if (ok and used_web) else 0

        if verbose:
            print(f"[{config}] {i}/{total} ok={int(ok)} tool_ok={int(used_web)} pred='{pred}' golds={golds[:3]} tool={tool}", flush=True)

        writer.writerow([
            run_id, config, i, masked, "; ".join(golds), pred, int(ok), round(dt, 3), (tool or ""), int(used_web)
        ])

        if PACE_S > 0:
            time.sleep(PACE_S)

    acc = (correct / total) if total else 0.0
    tool_acc = (tool_hits / total) if total else 0.0
    joint_acc = (joint_hits / total) if total else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    if verbose:
        print(f"[{config}] pred_acc={acc*100:.2f}%  tool_acc={tool_acc*100:.2f}%  joint_acc={joint_acc*100:.2f}%")

    return {
        "config": config,
        "n": total,
        "acc": acc,
        "tool_acc": tool_acc,
        "joint_acc": joint_acc,
        "avg_latency_s": avg_lat
    }

# -------------------------
# Main
# -------------------------
def main():
    _ensure_keys_loaded()

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["trex", "google_re", "conceptnet", "squad"],
                    help="LAMA configs to run")
    ap.add_argument("--n", type=int, default=200, help="Max examples per config (sampled)")
    ap.add_argument("--model", default=None, help="Groq model id (e.g., openai/gpt-oss-20b)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Controller temperature")
    ap.add_argument("--csv", default="logs/lama_agent_run.csv", help="CSV log file")
    ap.add_argument("--quiet", action="store_true", help="Less verbose progress")
    args = ap.parse_args()

    # Validate configs
    for c in args.configs:
        if c not in VALID_CONFIGS:
            raise ValueError(f"Unsupported config '{c}'. Choose from: {sorted(VALID_CONFIGS)}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    # Build controller
    agent = build_controller(model=args.model, temperature=args.temperature)

    run_id = f"lama_agent_{int(time.time())}"
    summary = []
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id","config","idx","masked_sentence","gold_variants",
            "prediction","correct","latency_s","tool_calls","tool_ok"
        ])
        for cfg in args.configs:
            stats = eval_config_with_agent(agent, cfg, args.n, writer, run_id, verbose=(not args.quiet))
            summary.append(stats)

    print("\n=== LAMA (Agent) Summary ===")
    for s in summary:
        print(
            f"{s['config']:>11}: n={s['n']:>5}  "
            f"pred_acc={s['acc']*100:6.2f}%  "
            f"tool_acc={s['tool_acc']*100:6.2f}%  "
            f"joint_acc={s['joint_acc']*100:6.2f}%  "
            f"avg_latency={s['avg_latency_s']:.2f}s"
        )
    print(f"CSV saved to: {args.csv}")

if __name__ == "__main__":
    main()
