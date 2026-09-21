"""
Embed the bilingual corpus using BGE-M3 and build a FAISS index for
cross-lingual retrieval.

Reads:  data/processed/corpus.jsonl
Writes: data/processed/faiss_index.bin
        data/processed/corpus_metadata.json

Usage:
    python src/retrieval/build_index.py
"""

import json
import os
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel


def load_corpus(path="data/processed/corpus.jsonl"):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_index(corpus_path="data/processed/corpus.jsonl",
                 index_path="data/processed/faiss_index.bin",
                 metadata_path="data/processed/corpus_metadata.json"):

    print("Loading corpus...")
    records = load_corpus(corpus_path)
    print(f"  {len(records)} documents loaded")

    print("\nLoading BGE-M3 model (downloads ~2GB the first time, be patient)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    texts = [r["text"] for r in records]

    print(f"\nEmbedding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=8,
        max_length=512,
    )["dense_vecs"]

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"\nIndex built: {index.ntotal} vectors, dimension {dim}")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved index -> {index_path}")
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    build_index()