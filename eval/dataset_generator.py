"""
eval/dataset_generator.py
==========================
Auto-generates 200 Q&A pairs using Groq for RAGAS-style evaluation.

Strategy:
- Sample chunks from each book and surah type
- Ask Groq to generate a question + reference answer from each chunk
- Store in eval_qa_pairs table and as a CSV

Usage:
    python -m eval.dataset_generator --n 200
    python -m eval.dataset_generator --n 50 --surah 1   # quick test on Al-Fatiha
"""

import sys
import csv
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATABASE_URL, DATA_PROC_DIR
from rag.generator import create_groq_client, LLM_MODEL

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

QUESTION_TYPES = ["factual", "explanatory", "comparative"]
DIFFICULTIES    = ["easy", "medium", "hard"]

GENERATION_PROMPT = """أنت خبير في علوم القرآن الكريم. بناءً على مقطع التفسير التالي، اصنع سؤالاً وإجابةً مرجعية.

مقطع التفسير:
سورة: {surah_name_ar} | الآية: {ayah_number} | المصدر: {book_name_ar}
نص الآية: {ayah_text}
التفسير: {tafseer_text}

اصنع:
- سؤالاً من نوع: {question_type} (سؤال {difficulty_desc})
- إجابة مرجعية مفصلة بالعربية مستندة إلى النص

أجب بتنسيق JSON فقط (بدون أي نص إضافي):
{{
  "question": "...",
  "reference_answer": "...",
  "question_type": "{question_type}",
  "difficulty": "{difficulty}"
}}"""

DIFFICULTY_DESC = {
    "easy":   "مباشر وبسيط يمكن الإجابة عليه من الآية مباشرة",
    "medium": "يتطلب فهم التفسير والسياق",
    "hard":   "يتطلب تحليلاً عميقاً أو مقارنة بين مفاهيم"
}


def sample_chunks(conn, n: int, surah_filter: Optional[int] = None) -> list[dict]:
    """Sample n diverse chunks from the DB for question generation."""
    where = "WHERE surah_number = %(surah)s" if surah_filter else ""
    params = {"surah": surah_filter, "n": n * 2}  # over-sample, then deduplicate

    sql = f"""
    SELECT chunk_id, book_name_ar, surah_name_ar, surah_number,
           ayah_number_start, ayah_text, tafseer_text
    FROM tafseer_chunks
    {where}
    ORDER BY RANDOM()
    LIMIT %(n)s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

    # Deduplicate by (surah, ayah) — prefer variety over repetition
    seen   = set()
    unique = []
    for r in rows:
        key = (r["surah_number"], r["ayah_number_start"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:n]


def generate_qa_for_chunk(client, chunk: dict) -> Optional[dict]:
    """Ask Groq to generate one Q&A pair for the given chunk."""
    q_type     = random.choice(QUESTION_TYPES)
    difficulty = random.choice(DIFFICULTIES)

    prompt = GENERATION_PROMPT.format(
        surah_name_ar=chunk["surah_name_ar"],
        ayah_number=chunk["ayah_number_start"],
        book_name_ar=chunk["book_name_ar"],
        ayah_text=chunk["ayah_text"],
        tafseer_text=chunk["tafseer_text"][:800],  # limit to avoid token overflow
        question_type=q_type,
        difficulty=difficulty,
        difficulty_desc=DIFFICULTY_DESC[difficulty],
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()

        # Strip any markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        data["reference_chunk_ids"] = [chunk["chunk_id"]]
        data["surah_number"]        = chunk["surah_number"]
        return data

    except (json.JSONDecodeError, KeyError, Exception) as e:
        log.warning(f"Failed to generate QA for chunk {chunk['chunk_id']}: {e}")
        return None


def save_qa_to_db(conn, qa_pairs: list[dict]):
    sql = """
    INSERT INTO eval_qa_pairs
        (question, reference_answer, reference_chunk_ids, surah_number, difficulty, question_type)
    VALUES
        (%(question)s, %(reference_answer)s, %(reference_chunk_ids)s,
         %(surah_number)s, %(difficulty)s, %(question_type)s)
    ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, qa_pairs, page_size=20)
    conn.commit()


def save_qa_to_csv(qa_pairs: list[dict]):
    path = DATA_PROC_DIR / "eval_qa_pairs.csv"
    fieldnames = ["question", "reference_answer", "reference_chunk_ids",
                  "surah_number", "difficulty", "question_type"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for qa in qa_pairs:
            row = dict(qa)
            row["reference_chunk_ids"] = ";".join(row.get("reference_chunk_ids", []))
            writer.writerow(row)
    log.info(f"📄 Eval CSV saved: {path}")


def run_generation(n: int = 200, surah_filter: Optional[int] = None):
    conn   = psycopg2.connect(DATABASE_URL)
    client = create_groq_client()

    log.info(f"🎯 Generating {n} Q&A pairs …")
    chunks = sample_chunks(conn, n, surah_filter)
    log.info(f"   Sampled {len(chunks)} diverse chunks")

    qa_pairs = []
    for i, chunk in enumerate(chunks, 1):
        log.info(f"  [{i:3d}/{len(chunks)}] Generating QA for "
                 f"{chunk['surah_name_ar']} / {chunk['book_name_ar']} …")

        qa = generate_qa_for_chunk(client, chunk)
        if qa:
            qa_pairs.append(qa)

        # Groq free tier: 14,400 req/day, 70K TPM
        # Stay well under: 1 req every 1.5s = 40 RPM
        time.sleep(1.5)

        # Batch-save every 20 items
        if len(qa_pairs) % 20 == 0 and qa_pairs:
            save_qa_to_db(conn, qa_pairs[-20:])
            log.info(f"   💾 Saved batch to DB ({len(qa_pairs)} total so far)")

    # Save remainder
    if qa_pairs:
        save_qa_to_db(conn, qa_pairs)

    save_qa_to_csv(qa_pairs)
    conn.close()

    log.info(f"\n✅ Generated {len(qa_pairs)}/{n} Q&A pairs")
    return qa_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=200, help="Number of Q&A pairs to generate")
    parser.add_argument("--surah", type=int, help="Restrict to one surah (for testing)")
    args = parser.parse_args()

    run_generation(n=args.n, surah_filter=args.surah)
