# src/eval/gsm8k_eval.py
import os, re, random, csv, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
from typing import Dict, Any, List

from datasets import load_dataset
from tqdm import tqdm
from tabulate import tabulate
from dotenv import load_dotenv

from src.tools.gsm8k_solver import gsm8k_solve_with_meta

# ---------- Config / env ----------
GSM8K_TIMEOUT_S = float(os.getenv("GSM8K_TIMEOUT_S", "25"))
PACE_S = float(os.getenv("GSM8K_PACE_S", "0.0"))  # small delay per item (seconds)

# ---------- Helpers ----------
def extract_gold(ans: str):
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", ans)
    return m.group(1) if m else None

def extract_num(ans: str):
    m = re.search(r"Answer:\s*([-+]?\d+(?:\.\d+)?)", ans)
    return m.group(1) if m else None

def _run_with_timeout(fn, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=GSM8K_TIMEOUT_S)

def _ensure_openai_key():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in your environment or in a .env file."
        )

# ---------- Main eval ----------
def main(split="test", n=100, seed=0, model=None, csv_path=None):
    _ensure_openai_key()

    random.seed(seed)
    ds = load_dataset("gsm8k", "main")[split]
    rows = ds.shuffle(seed=seed).select(range(n))

    pred_correct = 0
    tool_hits = 0
    joint_hits = 0

    table: List[List[str]] = []
    logs: List[Dict[str, Any]] = []

    for idx, r in enumerate(tqdm(rows, total=len(rows))):
        q = r["question"]
        gold = extract_gold(r["answer"]) or ""

        # --- Always call the gsm8k solver tool, with a watchdog timeout ---
        t0 = time.time()
        tool_ok = 0
        try:
            pred_full, meta = _run_with_timeout(gsm8k_solve_with_meta, q, model=model)
            tool_ok = 1  # tool call returned successfully
        except _TO:
            pred_full, meta = ("timeout", {"latency_s": None, "model": model})
        except Exception as e:
            pred_full, meta = (f"error: {e}", {"latency_s": None, "model": model})
        dt = time.time() - t0

        pred_num = extract_num(pred_full)
        ok = (pred_num == gold)

        pred_correct += int(ok)
        tool_hits += int(tool_ok)
        joint_hits += int(ok and tool_ok)

        table.append([
            q[:40] + ("..." if len(q) > 40 else ""),
            gold,
            pred_num or pred_full,
            "✅" if ok else "❌",
            "✅" if tool_ok else "❌",
            f"{dt:.2f}s"
        ])

        logs.append({
            "idx": idx,
            "gold": gold,
            "pred": pred_num or pred_full,
            "ok": int(ok),
            "tool_ok": int(tool_ok),
            "latency_s": f"{dt:.3f}",
            "reported_latency_s": f"{meta.get('latency_s', ''):.3f}" if meta.get("latency_s") is not None else "",
            "total_tokens": meta.get("total_tokens", ""),
            "prompt_tokens": meta.get("prompt_tokens", ""),
            "completion_tokens": meta.get("completion_tokens", ""),
            "model": meta.get("model", model or ""),
            "question": q.replace("\n", " ")[:1000],
        })

        if PACE_S > 0:
            time.sleep(PACE_S)

    pred_acc = pred_correct / len(rows)
    tool_acc = tool_hits / len(rows)
    joint_acc = joint_hits / len(rows)

    print(tabulate(
        table,
        headers=["question", "gold", "pred", "pred_ok", "tool_ok", "latency"],
        tablefmt="github"
    ))
    print(
        f"\nResults (n={len(rows)}, model={model or os.getenv('MODEL','env.MODEL')})\n"
        f" - pred_acc : {pred_correct}/{len(rows)} = {pred_acc:.2%}\n"
        f" - tool_acc : {tool_hits}/{len(rows)} = {tool_acc:.2%}\n"
        f" - joint_acc: {joint_hits}/{len(rows)} = {joint_acc:.2%}\n"
    )

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fieldnames = list(logs[0].keys()) if logs else [
            "idx","gold","pred","ok","tool_ok","latency_s","reported_latency_s",
            "total_tokens","prompt_tokens","completion_tokens","model","question"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(logs)
        print(f"Saved CSV logs to {csv_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)  # default to 100 examples
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="OpenAI model id, e.g. gpt-4o-mini")
    ap.add_argument("--csv", dest="csv_path", default=None, help="CSV output path e.g., logs/gsm8k_run.csv")
    args = ap.parse_args()
    main(**vars(args))
