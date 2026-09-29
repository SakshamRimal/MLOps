import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.retriever import retrieve, format_context
from app.rag.vectorstore import vector_store
from app.llm_client import llm_client


async def main():
    query = "Based on the knowledge base, what is decorator and generator in python?"
    print(f"Testing RAG retrieval for query: '{query}'")
    print(f"Total chunks in vector store: {vector_store.count()}")

    chunks = retrieve(query, top_k=2)
    print(f"Retrieved chunks count: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f" Chunk {i+1} [distance {c['distance']:.3f}]: {c['text'][:120]}...")

    context = format_context(chunks)
    print(f"\nFormatted context length: {len(context)} chars")

    try:
        result = await llm_client.chat_structured(query, context=context)
        print("\nRAG Structured Output:")
        print("Answer:", result.get("answer"))
        print("Sources:", result.get("sources"))
        print("Confidence:", result.get("confidence"))
    except Exception as e:
        print(f"\nLLM generation raised (expected if offline): {e}")


if __name__ == "__main__":
    result = llm_client.chat_with_tools(
        "Based on the knowledge base, what is decorator and generator in python? Please provide a brief explanation and an example for each."
    )
    print("Answer:", result["answer"])
    print("Tools used:", result["tool_calls_made"])
    asyncio.run(main())