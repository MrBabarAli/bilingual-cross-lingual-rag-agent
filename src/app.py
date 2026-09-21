"""
Streamlit demo UI for the Bilingual Cross-Border RAG + Agent System.

Usage:
    streamlit run src/app.py
"""

import os
import sys
import json
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from FlagEmbedding import BGEM3FlagModel
import time

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"

st.set_page_config(
    page_title="Bilingual Cross-Lingual RAG Agent",
    page_icon="🌐",
    layout="wide",
)


# ---------- Cached resources (loaded once) ----------

@st.cache_resource
def load_embedding_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)


@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("data/processed/faiss_index.bin")
    with open("data/processed/corpus_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource
def load_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
        st.stop()
    return genai.Client(api_key=api_key)


# ---------- Core pipeline functions ----------

def retrieve(query, embed_model, index, metadata, top_k=6):
    query_vec = embed_model.encode([query])["dense_vecs"]
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
            "text": doc["text"][:800],
            "doc_id": doc["doc_id"],
        })
    return results


def format_sources_for_prompt(sources):
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"[{i}] ({s['language'].upper()}) {s['title']}\n{s['text']}")
    return "\n\n".join(lines)


def call_gemini_with_retry(client, prompt, max_retries=3, delay=5):
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    return f"(Agent call failed after retries: {last_err})"


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
    return call_gemini_with_retry(client, prompt)


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
    return call_gemini_with_retry(client, prompt)


# ---------- UI ----------

st.title("🌐 Bilingual Cross-Border RAG & Agent System")
st.caption(
    "Ask a question in English or Chinese. The system retrieves relevant sources "
    "in BOTH languages using a shared multilingual embedding space (BGE-M3), then "
    "synthesizes a cited answer and checks for cross-lingual conflicts."
)

with st.sidebar:
    st.header("About this system")
    st.markdown(
        """
        **Pipeline:**
        1. **Retrieval** — BGE-M3 embeddings + FAISS, searching 71 bilingual
           documents (41 English arXiv papers, 30 Chinese Wikipedia articles)
        2. **Synthesis Agent** — generates a cited bilingual answer (Gemini)
        3. **Conflict-Detection Agent** — checks whether EN and ZH sources
           actually agree

        **Novelty:** cross-lingual retrieval without translation, plus explicit
        conflict detection between language communities' framing of a topic.
        """
    )
    top_k = st.slider("Number of sources to retrieve", min_value=3, max_value=10, value=6)

question = st.text_input(
    "Your question (English or Chinese):",
    placeholder="e.g. how do large language models work / 强化学习是什么",
)

run_button = st.button("Run query", type="primary")

if run_button and question.strip():
    with st.spinner("Loading models (first run may take a minute)..."):
        embed_model = load_embedding_model()
        index, metadata = load_index_and_metadata()
        client = load_gemini_client()

    with st.spinner("Retrieving bilingual sources..."):
        sources = retrieve(question, embed_model, index, metadata, top_k)

    st.subheader("📚 Retrieved sources")
    cols = st.columns(2)
    en_sources = [s for s in sources if s["language"] == "en"]
    zh_sources = [s for s in sources if s["language"] == "zh"]

    with cols[0]:
        st.markdown(f"**English ({len(en_sources)})**")
        for s in en_sources:
            with st.expander(f"{s['title'][:60]}... (score: {s['score']:.3f})"):
                st.write(s["text"])

    with cols[1]:
        st.markdown(f"**Chinese ({len(zh_sources)})**")
        for s in zh_sources:
            with st.expander(f"{s['title']} (score: {s['score']:.3f})"):
                st.write(s["text"])

    with st.spinner("Running synthesis agent..."):
        answer = synthesis_agent(client, question, sources)

    st.subheader("🤖 Synthesized bilingual answer")
    st.markdown(answer)

    with st.spinner("Running conflict-detection agent..."):
        conflict_report = conflict_agent(client, question, sources)

    st.subheader("⚖️ Cross-lingual conflict check")
    st.info(conflict_report)

elif run_button:
    st.warning("Please enter a question first.")