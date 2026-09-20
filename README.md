# CloudSense AI — Step 14: Semantic RAG + Evidence-Grounded Agent

Adds semantic RAG to Step 13 while preserving a deterministic fallback.

## What changed
- OpenAI `text-embedding-3-small` retrieval when `OPENAI_API_KEY` is configured.
- Local cached document embeddings in `knowledge_base/.embedding_cache.json`.
- Lexical retrieval fallback when embeddings are unavailable.
- RAG results include score, retrieval method, and evidence IDs such as `[rightsizing_policy.txt#0]`.
- Agent policy tool now returns evidence IDs for grounded citations.
- New `GET /api/rag/status` endpoint reports RAG mode and index size.
- Existing `/api/rag/search`, `/api/rag/documents`, AWS tools, ML tools, and Step 13 tool-calling agent remain intact.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
uvicorn app.main:app --reload
```

If `OPENAI_API_KEY` is omitted, the system still works using lexical retrieval.

## Demo question
`Which production EC2 resources can I right-size, and what approval is required?`

The agent should combine AWS resource evidence with policy evidence and cite the returned policy chunk IDs.
