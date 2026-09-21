"""
Collect Chinese-language NLP/LLM content.

Two modes:
1. --mode wiki   : Pull articles from Chinese Wikipedia via the official API
                   (reliable, legal, no scraping needed).
2. --mode manual : Interactively paste in content you've copied from sources
                   like 机器之心 (JiqiZhixin) or 量子位 (QbitAI).

Usage:
    python collect_zh_sources.py --mode wiki
    python collect_zh_sources.py --mode manual
"""

import requests
import json
import os
import argparse
import re
import time

WIKI_API = "https://zh.wikipedia.org/w/api.php"

DEFAULT_TOPICS = [
    "大型语言模型",       # Large language models
    "检索增强生成",       # Retrieval-augmented generation
    "自然语言处理",       # NLP
    "提示工程",           # Prompt engineering
    "机器翻译",           # Machine translation
    "深度学习",           # Deep learning
    "变换器模型",         # Transformer model
    "大模型幻觉",         # LLM hallucination
]


def slugify(text, max_len=40):
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s]+", "_", text)
    return text[:max_len]


def fetch_wiki_article(title):
    """Fetch plaintext extract of a Chinese Wikipedia article."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "redirects": 1,
    }
    headers = {"User-Agent": "BilingualRAGResearch/1.0 (student research project)"}
    resp = requests.get(WIKI_API, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "extract" in page and page["extract"].strip():
            return page.get("title", title), page["extract"].strip()
    return None, None


def collect_wiki(topics, out_dir="data/raw/zh"):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for topic in topics:
        print(f"Fetching Chinese Wikipedia: {topic}")
        try:
            title, text = fetch_wiki_article(topic)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        if not text:
            print(f"  No article found for '{topic}', skipping.")
            continue

        record = {
            "id": f"zhwiki_{slugify(title)}",
            "title": title,
            "abstract": text[:3000],
            "language": "zh",
            "source": "zh.wikipedia.org",
            "topic_query": topic,
        }

        fname = f"{slugify(title)}.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        count += 1
        print(f"  Saved: {title} ({len(text)} chars)")
        time.sleep(0.5)

    print(f"\nDone. Collected {count} Chinese Wikipedia articles -> {out_dir}")


def collect_manual(out_dir="data/raw/zh"):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    print("Manual entry mode. Type 'done' as title to finish.\n")

    while True:
        title = input("Article title (Chinese): ").strip()
        if title.lower() == "done":
            break
        source = input("Source (e.g. jiqizhixin.com, qbitai.com): ").strip()
        url = input("URL: ").strip()
        print("Paste body text, then type END on its own line:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        body = "\n".join(lines).strip()

        if not body:
            print("Empty body, skipping.\n")
            continue

        record = {
            "id": f"manual_{slugify(title)}",
            "title": title,
            "abstract": body[:3000],
            "language": "zh",
            "source": source,
            "url": url,
        }
        fname = f"{slugify(title)}.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        count += 1
        print(f"Saved: {title}\n")

    print(f"Done. Collected {count} manually-entered Chinese articles -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["wiki", "manual"], default="wiki")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--out_dir", type=str, default="data/raw/zh")
    args = parser.parse_args()

    if args.mode == "wiki":
        collect_wiki(args.topics, args.out_dir)
    else:
        collect_manual(args.out_dir)