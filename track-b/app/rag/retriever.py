from app.rag.vectorstore import vector_store


def retrieve(query: str, top_k: int = 4, max_distance: float = 0.8) -> list[dict]:
    """
    Retrieves relevant chunks for a query.
    Filters out chunks that are too dissimilar (distance too high) to avoid
    injecting irrelevant context into the prompt.
    """
    results = vector_store.query(query, top_k=top_k)
    return [r for r in results if r["distance"] <= max_distance]


def format_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a labeled string block for the prompt."""
    if not chunks:
        return ""

    parts = []
    for c in chunks:
        source = c["metadata"]["source"]
        parts.append(f"[Source: {source} | chunk_id: {c['chunk_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)