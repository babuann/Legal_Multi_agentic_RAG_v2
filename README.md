# Legal_Multi_agentic_RAG

A RAG system for querying U.S. securities-law documents using two agentic flows: **LangGraph** and **DeepAgents**.

---

## Architecture

```
User Query
    |
RouterAgent (Gemini Flash, T=0)
    | classifies: case_analysis / legal_argument / factual_lookup / comparative
    |
RetrieverAgent (ChromaDB + legal-contrastive embeddings)
    | adaptive k by route type
    |
SynthesizerAgent (Gemini Flash)
    | route-aware prompt, inline citations [N]
    |
ValidatorAgent (Gemini Flash, T=0)
    | scores relevance / grounding / completeness
    |
    +-- all >= 0.7? --> Answer
    |
    +-- no --> retry with refined query (max 2x)
```

---

## Two Flows

**LangGraph** uses a formal StateGraph with typed state and conditional retry edges. Same input always takes the same path. Good for when you need to know exactly what happened and why.

**DeepAgents** breaks the query into sub-questions, researches each one, reflects on gaps, and loops until confident. More thorough but slower and harder to bound cost-wise.

```
LangGraph:
START -> route_query -> retrieve_docs -> synthesize -> validate -> END
                            ^                               |
                            |________ retry ________________|

DeepAgents:
query -> decompose -> research sub-questions -> reflect on gaps
                            ^                        |
                            |____ follow-up Qs _______|
                                                     |
                                             final synthesis
```

---

## Project Structure

```
├── app.py               # Streamlit UI
├── main.py              # CLI runner
├── ingest.py            # Run once to index PDFs
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/                # PDF source files
├── models/
│   ├── legal-contrastive/   # active embedding model
│   └── legal-tsdae/
├── chroma_db/           # vector store (volume mount)
└── src/
    ├── config.py
    ├── ingestion.py
    ├── vector_store.py
    ├── llm_factory.py
    ├── rate_limiter.py
    ├── langgraph_flow.py
    ├── deep_agents_flow.py
    └── agents/
        ├── router_agent.py
        ├── retriever_agent.py
        ├── synthesizer_agent.py
        └── validator_agent.py
```

---

## Setup

**Prerequisites:** Docker + Docker Compose + Google AI Studio API key

```bash
cp .env.example .env
# set GOOGLE_API_KEY in .env

docker compose build

# index documents (run once, or after data changes)
docker compose run --rm ingest

# start the app
docker compose up rag-pipeline
# open http://localhost:8501
```

**Without Docker:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
streamlit run app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | required | Gemini API key |
| `LLM_MODEL` | `gemini-2.5-flash` | model to use |
| `MAX_REQUESTS_PER_MINUTE` | `14` | rate limit |
| `CHUNK_SIZE` | `800` | chars per chunk |
| `CHUNK_OVERLAP` | `150` | overlap between chunks |
| `RETRIEVAL_K` | `5` | chunks to retrieve |
| `MAX_ITERATIONS` | `1` | DeepAgents loop limit |

Set `MAX_REQUESTS_PER_MINUTE=1` to test rate limiting.

---

## Why Ingestion is Separate

`ingest.py` runs once offline. The app only reads from the already-built vector store. Re-embedding on every query would be slow and pointless since the documents don't change. This way the app starts in seconds, and the vector store can be rebuilt independently if the data changes.

---

## What Each Agent Does

**RouterAgent** classifies the query into one of four types before retrieval. This matters because different questions need different amounts of context. A factual lookup needs k=5 chunks. A comparative question needs k=10 to cover multiple documents. Without routing you either pull too many chunks (adds noise) or too few (misses context).

**RetrieverAgent** fetches the closest chunks from ChromaDB with k scaled by route. It uses the fine-tuned `legal-contrastive` model so similarity scores reflect actual legal relationships, not just surface vocabulary.

**SynthesizerAgent** writes the answer using only the retrieved context. The prompt structure changes by route: `case_analysis` enforces Parties/Issue/Holding/Reasoning/Outcome, `comparative` enforces side-by-side layout. Every claim must include a citation `[N]`.

**ValidatorAgent** scores the answer on relevance, grounding, and completeness (0 to 1 each). If all three are above 0.7, done. If not, it generates a better-rephrased query and the pipeline retries.

---

## Design Decisions

**Chunking:** `RecursiveCharacterTextSplitter` at 800 chars with 150-char overlap and legal separators (`\n\n`, `\n`, `. `, `; `, `, `). It tries paragraph breaks first, then sentences, then clauses. 800 chars fits within the embedding model's 256-token window while keeping arguments intact.

**Embeddings:** `legal-contrastive` replaces the generic `all-MiniLM-L6-v2` baseline. Legal text has specialised vocabulary like `scienter`, `materiality`, and `Rule 10b-5(b)` that general models handle poorly. `legal-contrastive` was fine-tuned in two stages on this exact corpus: TSDAE (unsupervised domain adaptation) then MultipleNegativesRankingLoss contrastive training on 4,500 legal passage pairs. To switch models, only `EMBEDDING_MODEL` in config and a re-run of `ingest.py` need to change.

**ChromaDB:** Runs in-process, persists to disk, no separate service needed. The `VectorStore` class wraps everything so switching backends means changing one file.

**Rate limiter:** Sliding window over 60 seconds tracks actual timestamps instead of a fixed window, so it stays within the Gemini free tier quota (15 RPM) without burst spikes. A tenacity retry decorator handles `ResourceExhausted` errors on top of that.

---

## Known Limitations

- No reranker. Chunks are ranked by cosine similarity only. A cross-encoder reranker after retrieval would help precision on comparative queries.
- No eval framework. No automated way to measure retrieval quality or answer faithfulness. RAGAS would help here.
- Flat chunking. Parent-child chunking would let small chunks be retrieved for precision while passing the full parent to the LLM for context.
- DeepAgents confidence is heuristic. It uses answer length and chunk count, not actual faithfulness. Using the validator's grounding score would be more reliable.
- Single collection. All documents share one ChromaDB collection. Metadata filtering per document would prevent cross-contamination on document-scoped queries.

---

## Source Documents

| File | Description |
|---|---|
| `23-980_petbrief.pdf` | Facebook v. Amalgamated Bank, Petitioner's brief |
| `23-980bsacus_...pdf` | Facebook v. Amalgamated Bank, US Gov amicus brief |
| `22-1165_10n2.pdf` | Ninth Circuit opinion |
| `USCOURTS-ilnd-...pdf` | Last Atlantis Capital v. AGS, District Court opinion |
| `comp26286.pdf` | SEC complaint |

---

## Tech Stack

- LLM: Google Gemini 2.5 Flash via `langchain-google-genai`
- Embeddings: fine-tuned `sentence-transformers`, CPU
- Vector DB: ChromaDB
- Orchestration: LangGraph + custom DeepAgents loop
- UI: Streamlit
- Rate limiting: sliding-window + tenacity retry
