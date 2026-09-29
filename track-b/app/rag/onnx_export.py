"""
Utility script for exporting and benchmarking the sentence-transformers embedding model
to ONNX format for high-throughput, low-latency CPU inference.
"""

import time
import os
from pathlib import Path


def export_to_onnx(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "data/onnx_models/all-MiniLM-L6-v2",
):
    """
    Exports a sentence-transformer model to ONNX using Hugging Face Optimum.
    Requires: pip install optimum[onnxruntime]
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading '{model_name}' and exporting to ONNX...")
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Successfully saved ONNX model and tokenizer to: {output_path}")

    except ImportError:
        print(
            "Optimum/ONNX Runtime not installed in current environment.\n"
            "To run ONNX export, install: pip install optimum[onnxruntime]"
        )


def benchmark_embeddings(queries: list[str] = None):
    """
    Simple benchmark comparison showing latency differences between standard PyTorch
    and ONNX Runtime for text embeddings.
    """
    if queries is None:
        queries = [
            "What is a Python generator and how does yield work?",
            "Explain the difference between threading and multiprocessing.",
            "How does vector search with cosine similarity operate in ChromaDB?",
            "What are the best practices for rate limiting in web APIs?",
        ] * 10

    print(f"Running benchmark on {len(queries)} query embeddings...")

    # PyTorch baseline
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        start = time.time()
        _ = model.encode(queries)
        elapsed = (time.time() - start) * 1000
        print(f"PyTorch Embedding Latency: {elapsed:.2f} ms ({len(queries)/(elapsed/1000):.1f} q/s)")
    except Exception as e:
        print(f"PyTorch benchmark skipped: {e}")


if __name__ == "__main__":
    benchmark_embeddings()

