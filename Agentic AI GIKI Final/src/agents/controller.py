from agno.agent import Agent
from agno.models.openai import OpenAIChat
import os
from src.tools.web_search import web_search
from src.tools.calculator import calc
from src.tools.gsm8k_solver import gsm8k_solve
from src.tools.rag import rag_query, rag_solve_math_from_docs

CONTROLLER_SYSTEM = (
    "You are a controller that must choose the correct tool and NOT guess.\n"
    "\n"
    "Routing rules (evaluate in order):\n"
    "A) Solve math problems from local PDFs → rag_solve_math_from_docs:\n"
    "   - If the user asks to solve/answer math word problems from local documents or PDFs,\n"
    "     call rag_solve_math_from_docs. If the user provides a folder (e.g., 'folder=bootcamp_docs'), pass it.\n"
    "\n"
    "B) General search in local PDFs → rag_query:\n"
    "   - If the user asks to find, quote, or summarize content from local PDFs (not specifically math problems),\n"
    "     call rag_query (with folder if provided).\n"
    "\n"
    "C) Public facts/news/entities → web_search:\n"
    "   - If the user’s question is about live facts, news, or public information, call web_search.\n"
    "\n"
    "D) Pure arithmetic → calc:\n"
    "   - Only if the input is a plain numeric expression with digits and operators (+ - * / ** % ( ) .) and no words.\n"
    "   - Examples: '2+2', '(120-35)/5', '18*(5+7)', '2**10'.\n"
    "\n"
    "E) Natural-language math word problems → gsm8k_solve:\n"
    "   - If the question is a written word problem (mentions people, items, money, time, units, etc.),\n"
    "     call gsm8k_solve.\n"
    "   - Examples: 'A class has 24 students, 4 per desk, how many desks?',\n"
    "     'A taxi charges 500 rupees per day + 20 per km; 3 days and 150 km, total cost?'\n"
    "\n"
    "You may call multiple tools in sequence if the user request contains multiple distinct tasks.\n"
    "For example:\n"
    "- If the user asks for both local doc info and a live web fact, call rag_query first then web_search.\n"
    "- If the user also asks to compute an arithmetic expression, call calc as a separate step.\n"
    "- If the user includes a natural-language math word problem, call gsm8k_solve for that part.\n"
    "Always return the combined results clearly.\n"
    "If the user prompt contains any inline numeric expression (like '56+98-100*0' or '2**10'), always call calc for that part, even if other tools are also required.\n"
    "\n"
    '''Folder handling rule:
- Never ask the user for a folder name for local PDFs/documents.
- Always assume the folder is os.getenv("RAG_FOLDER", "data") and call the tool immediately.
- Only use a provided 'folder=' argument if the user explicitly includes it (e.g., "folder=bootcamp_docs").
'''
    
" LAMA-style [MASK] completion → web_search:\n"
"   - If the user text contains the literal token '[MASK]', treat it as a cloze question.\n"
"   - Build a search query by removing the token and using the surrounding sentence (up to ~25 words),\n"
"     and also try a variant that swaps '[MASK]' with a blank and with a wildcard.\n"
"   - Call web_search with the most complete variant.\n"
"   - From the returned titles/snippets, pick a SINGLE WORD that best fills the blank.\n"
"     Tie-breakers in order:\n"
"       1) Proper noun that appears immediately after nearby commas/prepositions (e.g., 'Tokyo, Japan' → 'Japan').\n"
"       2) Highest-frequency single word across snippets that co-occurs with the nearby anchor words\n"
"          (e.g., the 3–5 tokens before/after the mask in the prompt).\n"
"       3) If a multi-word candidate (e.g., 'New Zealand') is most likely, return its head word only ('New' is wrong → return 'NewZealand' is wrong; prefer the canonical single token—\n"
"          for country/city names, return the standard single-token form if commonly written as one word, else return the core word ('Japan', 'Wales', 'Victoria')).\n"
"   - Do NOT add explanations or punctuation—return just the chosen word.\n"
"\n"

    "Output rules:\n"
    "- Always return plain text (no markdown, latex, or code blocks).\n"
    "- For calc: output only the number (e.g., 216).\n"
    "- For gsm8k_solve: output 2–4 short 'Reasoning:' steps followed by 'Answer: <number>'.\n"
    "- If a RAG tool indicates missing PDFs or no matches, surface that message directly."

"- For [MASK] completion: return ONLY the single word that fills the blank (no quotes, no punctuation, no extra text).\n"
"- Prefer the canonical proper-noun capitalization (e.g., 'Japan', 'Wales', 'Victoria').\n"
"- If uncertain, still choose the highest-evidence single word; do not answer with 'unknown' or explanations.\n"

)
import re
def _one_token(s: str) -> str:
    # keep only the first alphanumeric/proper-noun-looking token
    m = re.search(r"[A-Za-z][A-Za-z\-’']*", s or "")
    return m.group(0) if m else (s or "").strip().split()[0] if s else ""


def build_controller(model: str | None = None, temperature: float = 0.2):
    chat = OpenAIChat(
        id=model or os.getenv("MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=temperature,
    )

    return Agent(
        model=chat,
        instructions=CONTROLLER_SYSTEM,
        tools=[web_search, rag_solve_math_from_docs, rag_query, calc, gsm8k_solve],
    )

