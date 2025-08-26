# 🤖 Tool-Calling Agent (GIKI AI Bootcamp)

**Authors:** Muhammad Abdul Saboor & Talha Asif  
**Bootcamp:** GIKI Full Stack AI Bootcamp by SkyElectric  

---

## 📌 Project Overview
This project demonstrates an **Agentic AI system** built using **Agno** and **OpenAI models**, capable of solving diverse tasks by intelligently routing queries to the right tools.  

At its core:  
- **1 Controller Agent**  
- **4 Tools**  
  - 🌐 **Web Search** (DuckDuckGo + fallbacks/timeouts)  
  - 🧮 **Calculator** (NumExpr for pure arithmetic)  
  - 📘 **GSM8K Solver** (word problem solver with concise reasoning steps)  
  - 📄 **RAG** (retrieval from PDF, DOCX, and TXT documents)  

The agent is designed to **combine multiple tools in one query** — for example:  
> *“From documents, solve math problems, calculate (120-35)/5, and tell me Islamabad’s weather.”*  
This will invoke **RAG → GSM8K Solver → Calculator → Web Search** sequentially and merge results.  

---

## ⚙️ Frameworks & Libraries
- **Agno** → lightweight agent framework (preferred over LangChain/LangGraph for deterministic tool-calling).  
- **OpenAI GPT-4o-mini** → controller LLM for routing.  
- **Streamlit** → interactive UI with memory (`st.session_state`).  
- **ChromaDB** → vector store for retrieval.  
- **NumExpr** → fast arithmetic evaluation.  
- **DDGS** → DuckDuckGo search API.  
- **pypdf / python-docx** → document parsing.  

---

## 🚀 Features
- ✅ Deterministic **controller prompt** with explicit routing rules.  
- ✅ Upload and query **PDF/DOCX/TXT documents**.  
- ✅ Support for **multi-tool workflows**.  
- ✅ Robust **fallbacks + timeouts** for web search.  
- ✅ Interactive **Streamlit UI** with chat history + document management.  

---

## 📊 Benchmarks
### LAMA Benchmark
- **Tool accuracy (controller):** 100%  
- **Result accuracy:**  
  - TREX: **85%**  
  - Google_RE: **70%**

### GSM8K Benchmark
- **Tool accuracy:** 100%  
- **Result accuracy:** 88%  

---

## 🖥️ Streamlit UI
- Drag & drop **PDF/DOCX/TXT** into the app.  
- Chat with the agent.  
- Session memory powered by `st.session_state`.  

---

## ▶️ How to Run
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Set your **OpenAI API key**:
```bash
export OPENAI_API_KEY=your_api_key_here
```

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

---

## 📂 Repo Structure
```
├── src/
│   ├── agents/          # Controller agent
│   ├── tools/           # Web, Calc, GSM8K Solver, RAG
│   └── eval/            # Benchmark scripts (LAMA, GSM8K)
├── streamlit_app.py     # Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🌟 Industry Relevance
- Enterprise **document Q&A** systems.  
- AI-powered **decision support**.  
- Generalizable **multi-tool orchestration** with robust fallbacks.  

---

## 📬 Acknowledgments
This project was created during the **GIKI Full Stack AI Bootcamp**, with the mentorship and guidance of the GIKI faculty.  
