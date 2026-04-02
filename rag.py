"""
RAG (Retrieval-Augmented Generation) engine for FAQ chatbot.

Parses an FAQ document, embeds Q&A pairs using OpenAI embeddings,
and retrieves the most relevant answers via cosine similarity.
"""

import re
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_FILE = "embeddings_cache.json"


def parse_faq(filepath: str) -> list[dict]:
    """Parse a markdown FAQ file into a list of {question, answer} dicts."""
    text = Path(filepath).read_text(encoding="utf-8")
    entries = []

    # Match bold **Q: ...** followed by A: ... until the next Q or section
    pattern = re.compile(
        r"\*\*Q:\s*(.+?)\*\*\s*\nA:\s*(.+?)(?=\n\n|\Z)", re.DOTALL
    )
    for match in pattern.finditer(text):
        question = match.group(1).strip()
        answer = match.group(2).strip()
        entries.append({"question": question, "answer": answer})

    if not entries:
        raise ValueError(f"No FAQ entries found in {filepath}. Check the file format.")

    return entries


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class FAQRetriever:
    def __init__(self, faq_path: str, client: OpenAI):
        self.client = client
        self.entries = parse_faq(faq_path)
        self.embeddings = self._load_or_build_embeddings(faq_path)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def _load_or_build_embeddings(self, faq_path: str) -> np.ndarray:
        cache_path = Path(CACHE_FILE)
        faq_mtime = Path(faq_path).stat().st_mtime

        # Use cache if it exists and the FAQ file hasn't changed
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            if cache.get("faq_mtime") == faq_mtime and len(cache.get("embeddings", [])) == len(self.entries):
                print(f"Loaded {len(self.entries)} FAQ embeddings from cache.")
                return np.array(cache["embeddings"])

        print(f"Embedding {len(self.entries)} FAQ entries with OpenAI ({EMBEDDING_MODEL})...")
        texts = [f"{e['question']} {e['answer']}" for e in self.entries]
        embeddings = self._embed(texts)

        cache_path.write_text(json.dumps({"faq_mtime": faq_mtime, "embeddings": embeddings}))
        print("Embeddings cached to disk.")
        return np.array(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top_k most relevant FAQ entries for the query."""
        query_embedding = np.array(self._embed([query])[0])
        scores = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.entries[i], "score": scores[i]}
            for i in top_indices
        ]
