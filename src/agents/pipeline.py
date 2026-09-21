"""
Full agent pipeline: takes a user question (EN or ZH), retrieves bilingual
sources via the FAISS/BGE-M3 index, then runs two agents:

1. Synthesis Agent  - generates a cited, bilingual answer from retrieved sources
2. Conflict Agent   - checks whether the Chinese and English sources actually
                       agree, and flags any meaningful disagreement

Usage:
    python src/agents/pipeline.py "your question here"
"""

import os
import sys
import json
import argparse
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"


# ---------- Retrieval ----------

def load_metadata(path="data/processed/corpus_metadata.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(query, model, index, metadata, top_k=6):
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
            "text": doc["text"][:800],  # cap length fed to the LLM
            "doc_id": doc["doc_id"],
        })
    return results


# ---------- Agents ----------

def format_sources_for_prompt(sources):
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"[{i}] ({s['language'].upper()}) {s['title']}\n{s['text']}")
    return "\n\n".join(lines)


def synthesis_agent(client, question, sources):
    source_block = format_sources_for_prompt(sources)

    prompt = f"""You are a bilingual research assistant. Answer the user's question using ONLY the sources below. Sources are a mix of English and Chinese documents.

Instructions:
- Write the answer in the same language as the question, but explicitly draw on both English and Chinese sources where relevant.
- Cite sources inline using [1], [2], etc. matching the numbers below.
- If the sources don't fully answer the question, say so honestly.
- Keep the answer concise (150-250 words).

SOURCES:
{source_block}

QUESTION: {question}

ANSWER:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def conflict_agent(client, question, sources):
    en_sources = [s for s in sources if s["language"] == "en"]
    zh_sources = [s for s in sources if s["language"] == "zh"]

    if not en_sources or not zh_sources:
        return "Not enough sources in both languages to check for conflicts."

    source_block = format_sources_for_prompt(sources)

    prompt = f"""You are a fact-checking agent comparing English and Chinese sources on the same topic.

Below are English and Chinese sources related to the question: "{question}"

Your task:
1. Identify any claims where the English and Chinese sources genuinely DISAGREE or emphasize different things (not just different wording of the same fact).
2. If there is no real disagreement, say clearly: "No significant conflicts detected between English and Chinese sources on this topic."
3. Be conservative - only flag REAL disagreements, not superficial differences in phrasing or depth of detail.

SOURCES:
{source_block}

Respond in 2-4 sentences."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


# ---------- Main pipeline ----------

def run_pipeline(question, top_k=6):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    client = genai.Client(api_key=api_key)

    print("Loading retrieval model and index...")
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    index = faiss.read_index("data/processed/faiss_index.bin")
    metadata = load_metadata()

    print(f"\nQuestion: {question}\n")
    print("Retrieving bilingual sources...")
    sources = retrieve(question, embed_model, index, metadata, top_k)

    for i, s in enumerate(sources, 1):
        print(f"  [{i}] [{s['language'].upper()}] {s['title']} (score: {s['score']:.3f})")

    print("\nRunning synthesis agent...")
    answer = synthesis_agent(client, question, sources)

    print("\nRunning conflict-detection agent...")
    conflict_report = conflict_agent(client, question, sources)

    print("\n" + "=" * 70)
    print("BILINGUAL ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("CONFLICT CHECK (Chinese vs English sources)")
    print("=" * 70)
    print(conflict_report)

    return {
        "question": question,
        "sources": sources,
        "answer": answer,
        "conflict_report": conflict_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()

    run_pipeline(args.question, args.top_k)