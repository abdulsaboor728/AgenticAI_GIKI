import re
import numexpr as ne

_ALLOWED = re.compile(r"^[0-9\.\s\+\-\*\/\%\(\)]+(\*\*[0-9\.\s\+\-\*\/\%\(\)]+)?$")

def calc(expression: str) -> str:
    """
    Evaluate ONLY a pure numeric expression with no letters.
    Allowed: digits, spaces, + - * / ** % ( ) .
    Return just the final number as text.
    """
    expr = (expression or "").strip()
    # normalize any accidental caret power if you decide to tolerate it:
    expr = expr.replace("^", "**")  # optional; delete this line if you want strictness
    if not _ALLOWED.match(expr):
        return "Calculator error: disallowed characters"
    try:
        value = ne.evaluate(expr)
        return str(value.item() if hasattr(value, "item") else value)
    except Exception as e:
        return f"Calculator error: {e}"
