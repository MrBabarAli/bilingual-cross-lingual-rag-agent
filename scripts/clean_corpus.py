"""
Clean and unify the raw EN + ZH corpus into a single normalized format
ready for embedding.

Reads:  data/raw/en/*.json, data/raw/zh/*.json
Writes: data/processed/corpus.jsonl   (one JSON object per line)

Usage:
    python clean_corpus.py
"""

import json
import os
import glob
import re
import argparse


def load_json_files(folder):
    records = []
    for path in glob.glob(os.path.join(folder, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"  Skipping {path}: {e}")
    return records


def basic_clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_en_record(raw, idx):
    title = basic_clean_text(raw.get("title", ""))
    abstract = basic_clean_text(raw.get("abstract", ""))
    text = f"{title}. {abstract}" if title else abstract

    return {
        "doc_id": raw.get("id", f"en_{idx}"),
        "language": "en",
        "title": title,
        "text": text,
        "source": raw.get("source", "unknown"),
        "url": raw.get("url"),
        "raw_metadata": raw,
    }


def normalize_zh_record(raw, idx):
    title = basic_clean_text(raw.get("title", ""))
    abstract = basic_clean_text(raw.get("abstract", ""))
    text = f"{title}。{abstract}" if title else abstract

    return {
        "doc_id": raw.get("id", f"zh_{idx}"),
        "language": "zh",
        "title": title,
        "text": text,
        "source": raw.get("source", "unknown"),
        "url": raw.get("url"),
        "raw_metadata": raw,
    }


def dedupe(records, min_text_len=30):
    seen_texts = set()
    cleaned = []
    for r in records:
        text = r["text"]
        if len(text) < min_text_len:
            continue
        key = text[:200]
        if key in seen_texts:
            continue
        seen_texts.add(key)
        cleaned.append(r)
    return cleaned


def build_corpus(en_dir="data/raw/en", zh_dir="data/raw/zh",
                  out_path="data/processed/corpus.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading English records from {en_dir} ...")
    en_raw = load_json_files(en_dir)
    print(f"  Found {len(en_raw)} raw English files")

    print(f"Loading Chinese records from {zh_dir} ...")
    zh_raw = load_json_files(zh_dir)
    print(f"  Found {len(zh_raw)} raw Chinese files")

    en_records = [normalize_en_record(r, i) for i, r in enumerate(en_raw)]
    zh_records = [normalize_zh_record(r, i) for i, r in enumerate(zh_raw)]

    all_records = dedupe(en_records + zh_records)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_en = sum(1 for r in all_records if r["language"] == "en")
    n_zh = sum(1 for r in all_records if r["language"] == "zh")

    print(f"\nDone. Wrote {len(all_records)} records -> {out_path}")
    print(f"  English: {n_en}")
    print(f"  Chinese: {n_zh}")

    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_dir", default="data/raw/en")
    parser.add_argument("--zh_dir", default="data/raw/zh")
    parser.add_argument("--out_path", default="data/processed/corpus.jsonl")
    args = parser.parse_args()

    build_corpus(args.en_dir, args.zh_dir, args.out_path)