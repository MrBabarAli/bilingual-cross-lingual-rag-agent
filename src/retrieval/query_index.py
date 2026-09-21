"""
Query the bilingual FAISS index built by build_index.py.
Test cross-lingual retrieval: an English query should surface relevant
Chinese documents, and vice versa.

Usage:
    python src/retrieval/query_index.py "your query here"
    python src/retrieval/query_index.py "large language models" --top_k 5
"""

import json
import argparse
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel


def load_metadata(path="data/processed/corpus_metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search(query, model, index, metadata, top_k=5):
    query_vec = model.encode([query])["dense_vecs"]
    query_vec = np.array(query_vec).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = metadata[idx]
        results.append({
            "score": float(score),
            "language": doc["language"],
            "title": doc["title"],
            "text_preview": doc["text"][:150],
            "doc_id": doc["doc_id"],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Search query, in English or Chinese")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_path", default="data/processed/faiss_index.bin")
    parser.add_argument("--metadata_path", default="data/processed/corpus_metadata.json")
    args = parser.parse_args()

    print("Loading model and index...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    index = faiss.read_index(args.index_path)
    metadata = load_metadata(args.metadata_path)

    print(f"\nQuery: {args.query}\n")
    results = search(args.query, model, index, metadata, args.top_k)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['language'].upper()}] (score: {r['score']:.4f}) {r['title']}")
        print(f"   {r['text_preview']}...")
        print()


if __name__ == "__main__":
    main()