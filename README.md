# AI Knowledge Worker — RAG Project

A Retrieval-Augmented Generation (RAG) system that answers questions grounded in a
personal knowledge base, built in three stages to learn embeddings, vector retrieval,
and grounded generation from first principles before using frameworks.

Uses Google's Gemini models (free tier), called through the `openai` Python library
pointed at Gemini's OpenAI-compatible endpoint.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-real-key-here" > .env   # get one free at aistudio.google.com/apikey
```

## The three stages

**Day 1 — `day1_toy_rag.py`**: RAG implemented by hand, no framework. Embeds whole
files, finds the closest match with a manual cosine-similarity loop. Built to
understand the raw mechanism before any framework hides it.

**Day 2 — `day2_build_vector_store.py` + `day2_visualize_embeddings.py`**: a real
ingestion pipeline — documents split into overlapping chunks with LangChain's text
splitter, embeddings stored in ChromaDB instead of an in-memory list, plus a t-SNE
visualization of the embedding space.

**Day 3 — the `rag/` package + `app.py`**: everything wired into one real,
production-shaped system — a retrieval+generation chain with conversation history,
served through a Gradio chat UI, structured as proper Python modules instead of
standalone scripts.

## Running the full system (Day 3)

```bash
python build_index.py   # builds the vector store (run once, or after changing docs)
python app.py            # launches the Gradio chat UI at a local URL
```

### Project structure

```
rag/
  config.py       — Gemini client, model names, paths, chunking/retrieval constants
  ingestion.py     — loads and chunks documents from data/raw_docs/
  vectorstore.py   — embedding (with retry-on-rate-limit) + ChromaDB build/query
  chain.py         — the actual RAG chain: retrieve -> build context -> generate,
                      with conversation history support
build_index.py       — CLI entrypoint to (re)build the vector store
app.py                — Gradio chat UI entrypoint
data/raw_docs/
  profile/    profile.md
  projects/   store_intelligence_system.md, retail_sales_analysis.md
  skills/     skills_and_learning.md
```

### How conversation history works

`app.py` passes Gradio's running conversation history straight into
`rag.chain.answer()`, which prepends it to the messages sent to the model (after the
system prompt, before the new context-augmented question). This is what lets a
follow-up like "what stack did it use?" correctly resolve "it" to whatever project was
discussed in the previous turn — the model sees the full conversation, not just the
latest message in isolation.

Retrieval itself, though, only ever uses the latest message as the search query — it
doesn't currently rewrite the query using conversation context. That's a known,
nameable limitation (sometimes called "query rewriting" or "query contextualization")
rather than something broken — a reasonable next improvement, not a bug.

## Knowledge base

```
data/raw_docs/
  profile/    profile.md                     — background, career targets, roles
  projects/   store_intelligence_system.md   — the computer vision project
  projects/   retail_sales_analysis.md       — the data analysis project
  skills/     skills_and_learning.md         — skills, roadmaps, resume notes
```

## Possible next steps

- Query rewriting: use conversation history to rephrase the retrieval query itself,
  not just the final prompt
- Metadata filtering: let the UI scope retrieval to one `doc_type` category
- Evaluation: a golden Q&A test set with retrieval metrics (MRR/NDCG) and an
  LLM-as-judge for answer quality
