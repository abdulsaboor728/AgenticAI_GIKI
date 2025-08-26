# src/tools/web_search.py
import hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from ddgs import DDGS

# simple on-disk cache to avoid repeated network calls during evals
_CACHE_DIR = os.getenv("LAMA_CACHE", ".lama_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _ckey(q: str, k: str) -> str:
    base = hashlib.sha1(k.encode()).hexdigest()
    return os.path.join(_CACHE_DIR, f"{base}.json")

def _fetch(query: str, max_results: int, days: int | None):
    with DDGS() as ddgs:
        kw = {"max_results": max_results}
        if days:  # you rarely need days for LAMA
            kw["timelimit"] = f"d{days}"
        return ddgs.text(query, **kw) or []

def web_search(query: str, max_results: int = 5, days: int | None = None) -> str:
    """
    Robust search with timeout, retries, and caching.
    Returns a plain-text list of lines: "- title: url\\n  snippet"
    """
    key = _ckey("v1", f"{query}|{max_results}|{days}")
    # cache read
    if os.path.exists(key):
        try:
            return json.load(open(key, "r"))["text"]
        except Exception:
            pass

    TIMEOUT_S = int(os.getenv("WEB_SEARCH_TIMEOUT_S", "12"))
    tries, delay = 0, 0.6
    last_err = None

    while tries < 3:
        tries += 1
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_fetch, query, max_results, days)
                results = fut.result(timeout=TIMEOUT_S)

            lines = []
            for r in results[:max_results]:
                title = (r.get("title") or "").strip()
                href  = (r.get("href")  or "").strip()
                body  = (r.get("body")  or "").strip()[:300]
                lines.append(f"- {title}: {href}\n  {body}")

            text = "\n".join(lines) if lines else "No results."
            try: json.dump({"text": text}, open(key, "w"))
            except Exception: pass
            return text

        except TimeoutError:
            last_err = "timeout"
        except Exception as e:
            last_err = str(e)

        time.sleep(delay)
        delay *= 1.7

    return f"No results. Error: {last_err or 'unknown'}"
