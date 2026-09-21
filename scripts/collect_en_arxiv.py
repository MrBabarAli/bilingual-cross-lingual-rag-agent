"""
Collect English-language NLP/LLM papers from arXiv.
Saves each paper as a JSON file with metadata + abstract.

Usage:
    python collect_en_arxiv.py --topic "large language models" --max_results 40
"""

import arxiv
import json
import os
import argparse
import re


def slugify(text, max_len=60):
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s]+", "_", text)
    return text[:max_len]


def collect_arxiv_papers(topics, max_results_per_topic=20, out_dir="data/raw/en"):
    os.makedirs(out_dir, exist_ok=True)
    client = arxiv.Client()
    seen_ids = set()
    count = 0

    for topic in topics:
        print(f"\nSearching arXiv for: '{topic}'")
        search = arxiv.Search(
            query=topic,
            max_results=max_results_per_topic,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in client.results(search):
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)

            record = {
                "id": result.entry_id,
                "title": result.title.strip(),
                "abstract": result.summary.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "published": str(result.published.date()),
                "categories": result.categories,
                "url": result.entry_id,
                "language": "en",
                "source": "arxiv",
                "topic_query": topic,
            }

            fname = f"{slugify(result.title)}.json"
            path = os.path.join(out_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            count += 1
            print(f"  Saved: {record['title'][:70]}...")

    print(f"\nDone. Collected {count} unique English papers -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "large language models",
            "retrieval augmented generation",
            "cross-lingual NLP",
            "LLM agents",
            "prompt engineering",
        ],
        help="List of search topics",
    )
    parser.add_argument("--max_results", type=int, default=10, help="Max results per topic")
    parser.add_argument("--out_dir", type=str, default="data/raw/en")
    args = parser.parse_args()

    collect_arxiv_papers(args.topics, args.max_results, args.out_dir)