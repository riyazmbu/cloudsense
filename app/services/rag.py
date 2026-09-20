from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"
CACHE_FILE = KB_DIR / ".embedding_cache.json"
DEFAULT_MODEL = os.getenv("CLOUDSENSE_EMBEDDING_MODEL", "text-embedding-3-small")


def load_documents() -> List[Dict[str, str]]:
    documents = []
    if not KB_DIR.exists():
        return documents
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.append({"source": path.name, "text": text})
    return documents


def chunk_text(text: str, chunk_size: int = 450) -> List[str]:
    words = text.split()
    chunks, current = [], []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_index() -> List[Dict[str, Any]]:
    chunks = []
    for doc in load_documents():
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })
    return chunks


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_score(question: str, text: str) -> float:
    q = _terms(question)
    d = _terms(text)
    if not q or not d:
        return 0.0
    # Jaccard-like normalized overlap, useful as a no-API fallback.
    return len(q & d) / max(1, len(q))


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_cache() -> Dict[str, Any]:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(data: Dict[str, Any]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _embedding_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def _get_embeddings(texts: List[str]) -> List[List[float]] | None:
    client = _embedding_client()
    if client is None:
        return None
    response = client.embeddings.create(model=DEFAULT_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _domain_boost(question: str, text: str) -> float:
    q = question.lower()
    t = text.lower()
    groups = [
        (['ec2', 'instance', 'right-size', 'rightsizing', 'compute optimizer'], ['ec2', 'rightsizing', 'compute optimizer']),
        (['rds', 'database', 'db load'], ['rds', 'database', 'db load']),
        (['ebs', 'volume', 'unattached'], ['ebs', 'volume', 'unattached']),
        (['nat', 'nat gateway', 'vpc endpoint', 'cross-az'], ['nat gateway', 'vpc', 'cross-az']),
        (['s3', 'bucket', 'lifecycle', 'storage class'], ['s3', 'lifecycle', 'storage class']),
        (['lambda', 'function', 'duration'], ['lambda', 'memory', 'duration']),
        (['cloudfront', 'cache', 'origin', 'cache hit'], ['cloudfront', 'cache hit', 'origin']),
        (['cost explorer', 'billing', 'resource cost', 'cur'], ['cost explorer', 'resource-level', 'cost and usage']),
    ]
    boost = 0.0
    for q_terms, d_terms in groups:
        if any(x in q for x in q_terms) and any(x in t for x in d_terms):
            boost += 0.12
    return boost

def retrieve(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    index = build_index()
    if not index:
        return []

    # Prefer semantic embeddings when an OpenAI key is configured.
    try:
        query_embedding = _get_embeddings([question])
        if query_embedding:
            cache = _load_cache()
            cached_vectors = cache.get("vectors", {})
            texts = [c["text"] for c in index]
            missing = [t for t in texts if t not in cached_vectors]
            if missing:
                vectors = _get_embeddings(missing)
                if vectors:
                    for text, vector in zip(missing, vectors):
                        cached_vectors[text] = vector
                    cache.update({"model": DEFAULT_MODEL, "vectors": cached_vectors})
                    _save_cache(cache)
            scored = []
            qv = query_embedding[0]
            for chunk in index:
                vector = cached_vectors.get(chunk["text"])
                if vector:
                    score = _cosine(qv, vector) + _domain_boost(question, chunk["text"])
                    scored.append((score, chunk))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                results = []
                for score, chunk in scored[:top_k]:
                    item = dict(chunk)
                    item.update({"score": round(score, 4), "retrieval": "embedding"})
                    results.append(item)
                return results
    except Exception:
        # Retrieval must remain available even if the embedding provider is unavailable.
        pass

    scored = [(_lexical_score(question, c["text"]) + _domain_boost(question, c["text"]), c) for c in index]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in scored[:top_k]:
        item = dict(chunk)
        item.update({"score": round(score, 4), "retrieval": "lexical_fallback"})
        results.append(item)
    return results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Evidence ID: {c['source']}#{c['chunk_id']} | Score: {c.get('score', 0)}]\n{c['text']}"
        for c in chunks
    )


def retrieval_status() -> Dict[str, Any]:
    return {
        "embedding_model": DEFAULT_MODEL,
        "embedding_enabled": bool(os.getenv("OPENAI_API_KEY") and OpenAI is not None),
        "document_count": len(load_documents()),
        "chunk_count": len(build_index()),
        "cache_exists": CACHE_FILE.exists(),
    }
