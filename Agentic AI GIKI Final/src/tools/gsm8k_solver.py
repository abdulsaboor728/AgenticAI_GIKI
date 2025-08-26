# src/tools/gsm8k_solver.py
import os, re, time
from typing import Optional, Tuple, Dict, Any
from openai import OpenAI

SYSTEM = (
    "You are a careful math specialist. Provide brief, verifiable reasoning (2–4 short steps), "
    "not inner monologue. Always end with 'Answer: <number>'."
)

_client: OpenAI | None = None
def _openai_client() -> OpenAI:
    """
    Build a single OpenAI client. Reads:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL (optional; defaults to api.openai.com)
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

def gsm8k_solve(problem: str, model: Optional[str] = None) -> str:
    """
    Solve NATURAL-LANGUAGE math word problems (not pure expressions).
    Output: 2–4 brief 'Reasoning:' steps then 'Answer: <number>'.
    """
    ans, _ = gsm8k_solve_with_meta(problem, model=model)
    return ans

def gsm8k_solve_with_meta(problem: str, model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (full_text, meta) where full_text includes the reasoning lines + final 'Answer: <number>'.
    """
    client = _openai_client()
    model_id = model or os.getenv("MODEL", "o4-mini")

    user = (
        "Solve the following math word problem. Show 2–4 concise steps labeled 'Reasoning:' "
        "that are sufficient to verify the result (no long chain-of-thought), then output the final line "
        "as: Answer: <number>\n\n"
        f"Problem: {problem}"
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        temperature=0.0,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user},
        ],
    )
    dt = time.time() - t0

    text = (resp.choices[0].message.content or "").strip()

    # Keep evaluator compatibility: extract "Answer: <number>"
    m = re.search(r"Answer:\s*([-+]?\d+(?:\.\d+)?)", text)
    extracted = m.group(0) if m else text

    usage = getattr(resp, "usage", None)
    meta = {
        "latency_s": dt,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "model": model_id,
    }
    return text, meta
