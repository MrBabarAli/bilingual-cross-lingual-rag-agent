# Bilingual Cross-Border RAG & Agent System

**A Retrieval-Augmented Generation (RAG) and multi-agent system for cross-lingual (Chinese-English) academic and industrial knowledge retrieval.**

Author: **Babar Ali**
GitHub: [@MrBabarAli](https://github.com/MrBabarAli)
Email: mralibabar@outlook.com

---

## 1. Motivation

Academic and industrial knowledge is fragmented across language boundaries. Chinese researchers and companies often struggle to discover relevant Western literature, and vice versa. Standard RAG systems are built for monolingual corpora — they don't handle cross-lingual retrieval well, and translation-first pipelines lose nuance, especially around domain-specific terminology.

This project is a working proof-of-concept for a system that:
- Retrieves relevant documents in **both Chinese and English** for a single query, in either language, using a **shared multilingual embedding space** (no machine translation step involved in retrieval)
- **Synthesizes** a bilingual, cited answer from both language sources using an LLM agent
- **Detects conflicts** — checks whether Chinese-language and English-language sources actually agree on a topic, or emphasize/report it differently

This is a proposal-stage research prototype, built to demonstrate the technical feasibility of the core idea ahead of a full research program.

## 2. System architecture

**Retrieval:** [BGE-M3](https://huggingface.co/BAAI/bge-m3) (BAAI), a multilingual embedding model, embeds both English and Chinese documents into a single shared vector space. This means a query in one language can retrieve semantically relevant documents in the other — without a separate translation step.

**Agents:** Built on Google's Gemini API. The Synthesis Agent reads the retrieved bilingual sources and produces a single cited answer that draws on both. The Conflict-Detection Agent separately checks whether the English and Chinese sources actually agree, or whether they emphasize different things — flagging genuine disagreement rather than superficial wording differences.

## 3. Corpus

| Language | Source | Count |
|---|---|---|
| English | arXiv (cs.CL papers: LLMs, RAG, cross-lingual NLP, agents, prompt engineering) | 41 |
| Chinese | Chinese Wikipedia (AI/ML/NLP core concepts) | 30 |
| **Total** | | **71** |

This is a small, hand-curated prototype corpus intended to prove the retrieval and agent mechanisms work — not a production-scale dataset. See "Future Work" below for scaling plans.

## 4. Example results

**Query:** *"reinforcement learning applications"*

Retrieval returned a near-even mix of English and Chinese sources, interleaved by relevance score — evidence of genuine cross-lingual semantic matching rather than one language dominating:

| Rank | Language | Score | Title |
|---|---|---|---|
| 1 | EN | 0.651 | Reinforcement Learning for improving LLMs' Catalan text simplification |
| 2 | ZH | 0.643 | 强化学习 (Reinforcement Learning) |
| 3 | ZH | 0.550 | 监督学习 (Supervised Learning) |
| 4 | EN | 0.540 | What Matters, When? Diagnosing Visuomotor Imitation Policies |

**Query:** *"how do large language models work"*

The top-ranked result (score 0.748) was the Chinese Wikipedia article on Large Language Models — ranked *above* several English arXiv papers on the same topic — demonstrating that the retrieval understands query meaning across languages rather than relying on lexical/keyword overlap.

The Synthesis Agent's answer for this query combined a Chinese-sourced explanation of transformer self-attention architecture with English-sourced findings on RL-based fine-tuning (GRPO) and cross-lingual consistency research — genuinely merging both language sources into one coherent, cited answer, not just concatenating them.

The Conflict-Detection Agent correctly identified no substantive disagreement between sources, and explained *why*: Chinese sources were foundational/definitional in nature, while English sources were specialized recent research extensions — a reasoned judgment rather than a blind "no conflict" response.

*(Additional example screenshots from the Streamlit demo are in `/docs/screenshots` — see the demo video linked below.)*

## 5. What's novel here

Most existing RAG systems either:
- Assume a monolingual corpus, or
- Translate everything into one language before retrieval, which loses nuance in domain-specific terminology

This system instead:
1. Uses a **shared multilingual embedding space** so retrieval works on meaning, not translated keywords
2. Adds an explicit **conflict-detection agent** — most RAG/agent systems silently pick one source's framing when sources disagree; this system surfaces disagreement instead

## 6. Setup and usage

### Requirements
- Python 3.11+
- A [Google Gemini API key](https://ai.google.dev/) (free tier is sufficient for this prototype)

### Installation

```bash
git clone https://github.com/MrBabarAli/bilingual-cross-lingual-rag-agent.git
cd bilingual-cross-lingual-rag-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

### Rebuilding the corpus (optional — corpus is already included in `data/`)

```bash
python scripts/collect_en_arxiv.py --max_results 10
python scripts/collect_zh_sources.py --mode wiki
python scripts/clean_corpus.py
```

### Building the retrieval index

The FAISS index itself is not committed to this repo (regenerable, and binary files don't belong in version control). Build it with:

```bash
python src/retrieval/build_index.py
```

### Running a query from the command line

```bash
python src/agents/pipeline.py "how do large language models work"
```

### Running the interactive demo

```bash
streamlit run src/app.py
```

## 7. Limitations (honest assessment)

- **Corpus size and composition asymmetry:** The Chinese corpus (Wikipedia, encyclopedic/definitional) and English corpus (arXiv, novel research contributions) differ structurally. This causes some queries — especially narrow, definitional Chinese-language ones — to retrieve weaker English matches, since there's no direct English equivalent document type. Confirmed in testing with the query "提示工程是什么" (what is prompt engineering).
- **Small scale:** 71 documents is sufficient to prove the retrieval/agent mechanism works, not to serve as a production knowledge base.
- **No custom fine-tuning:** Uses off-the-shelf BGE-M3 embeddings. Domain-specific terminology fine-tuning (e.g., for patent law or specific technical fields) is proposed future work, not yet implemented.
- **No patent/CNKI data:** CNKI has strict anti-scraping protections; this prototype uses only openly accessible sources (arXiv, Wikipedia).

## 8. Future work

- Scale corpus to hundreds/thousands of documents per language, across multiple domains
- Add terminology-aware re-ranking (e.g., BGE-reranker-v2-m3) fine-tuned on Chinese-English technical term pairs
- Incorporate patent data (CNIPA/USPTO) via official APIs rather than scraping
- Build a proper bilingual evaluation benchmark with human expert judgments
- Multi-agent decomposition for complex queries (separate literature, patent, and terminology sub-agents)

## 9. Intended research context

This prototype was built to support a research proposal for a Chinese Government Scholarship (CSC) application, targeting research in cross-lingual NLP and agentic AI systems. *(University and supervisor details to be added once confirmed.)*

## 10. License

MIT License — see `LICENSE` file.