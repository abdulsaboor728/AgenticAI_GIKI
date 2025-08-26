import argparse
import time
from dotenv import load_dotenv
from src.agents.controller import build_controller

def _as_text(resp) -> str:
    # Agno RunResponse has .content; fallback to str for safety
    txt = getattr(resp, "content", None)
    if isinstance(txt, str):
        return txt.strip()
    return str(resp).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Groq model")
    ap.add_argument("--temperature", type=float, default=0.2, help="Controller temperature")
    args = ap.parse_args()

    load_dotenv()
    agent = build_controller(model=args.model, temperature=args.temperature)
    print(f"🤖 Controller ready (model={args.model or 'env.GROQ_MODEL'}) — type 'exit' to quit.")

    try:
        while True:
            q = input("\nYou> ").strip()
            if not q or q.lower() in {"exit", "quit"}:
                break
            t0 = time.time()
            resp = agent.run(q)
            ans = _as_text(resp)

            # Enforce super-clean output (no markdown/latex remnants)
            # quick strip of common formatting artifacts:
            ans = ans.replace("$$", "").replace("$", "").replace("\\boxed{", "").replace("}", "")
            print("Agent>", ans)
            print(f"(answered in {time.time() - t0:.2f}s)")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")

if __name__ == "__main__":
    main()
