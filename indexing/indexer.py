"""
indexing/indexer.py
===================
Reads the master CSV → embeds all chunks → upserts into pgvector.

Run after data/collector.py has produced master_tafseer_dataset.csv.

Usage:
    python -m indexing.indexer                         # full index
    python -m indexing.indexer --book almuyassar       # one book only
    python -m indexing.indexer --resume                # skip already-indexed chunks
    python -m indexing.indexer --status                # show index stats
"""

import csv
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATABASE_URL, DATA_PROC_DIR, EMBEDDING_DIM
from indexing.embedder import embed_texts

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# pgvector adapter for Python lists → vector type
def adapt_vector(v):
    return AsIs(f"'{v}'::vector")

register_adapter(list, adapt_vector)


# ── Database helpers ───────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(DATABASE_URL)


def get_indexed_chunk_ids(conn) -> set[str]:
    """Return the set of chunk_ids already in the DB (for resume mode)."""
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM tafseer_chunks")
        return {row[0] for row in cur.fetchall()}


def get_index_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(embedding) AS with_embedding,
                COUNT(DISTINCT book_slug) AS books,
                COUNT(DISTINCT surah_number) AS surahs
            FROM tafseer_chunks
        """)
        row = cur.fetchone()
        return {
            "total":          row[0],
            "with_embedding": row[1],
            "books":          row[2],
            "surahs":         row[3],
        }


def upsert_chunks(conn, rows: list[dict], vectors: list[list[float]]):
    """Upsert a batch of rows with their embedding vectors."""
    sql = """
        INSERT INTO tafseer_chunks (
            chunk_id, book_api_id, book_slug, book_name_ar, book_name_en,
            author, surah_number, surah_name_ar, surah_name_en,
            revelation_type, ayah_number_start, ayah_number_end,
            juz, ayah_text, tafseer_text, text_for_embedding,
            word_count, char_count, embedding
        ) VALUES (
            %(chunk_id)s, %(book_api_id)s, %(book_slug)s, %(book_name_ar)s, %(book_name_en)s,
            %(author)s, %(surah_number)s, %(surah_name_ar)s, %(surah_name_en)s,
            %(revelation_type)s, %(ayah_number_start)s, %(ayah_number_end)s,
            %(juz)s, %(ayah_text)s, %(tafseer_text)s, %(text_for_embedding)s,
            %(word_count)s, %(char_count)s, %(embedding)s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            tafseer_text = EXCLUDED.tafseer_text,
            text_for_embedding = EXCLUDED.text_for_embedding
    """
    records = []
    for row, vec in zip(rows, vectors):
        r = dict(row)
        r["embedding"]       = str(vec)   # pgvector expects '[0.1, 0.2, ...]' string
        r["book_api_id"]     = int(r.get("book_api_id", 0))
        r["surah_number"]    = int(r.get("surah_number", 0))
        r["ayah_number_start"] = int(r.get("ayah_number_start", 0))
        r["ayah_number_end"]   = int(r.get("ayah_number_end", 0))
        r["juz"]             = int(r.get("juz", 1))
        r["word_count"]      = int(r.get("word_count", 0))
        r["char_count"]      = int(r.get("char_count", 0))
        records.append(r)

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, records, page_size=50)
    conn.commit()


# ── Main indexing pipeline ─────────────────────────────────────────────────────
def run_indexing(
    book_filter: Optional[str] = None,
    resume: bool = False,
    embed_batch_size: int = 8,
):
    master_csv = DATA_PROC_DIR / "master_tafseer_dataset.csv"
    if not master_csv.exists():
        log.error(f"Master CSV not found: {master_csv}")
        log.error("Run 'python -m data.collector' first.")
        sys.exit(1)

    # Load CSV
    log.info(f"📂 Loading {master_csv} …")
    with open(master_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    log.info(f"   Loaded {len(all_rows):,} rows")

    # Apply book filter
    if book_filter:
        all_rows = [r for r in all_rows if r["book_slug"] == book_filter]
        log.info(f"   Filtered to book '{book_filter}': {len(all_rows):,} rows")

    # Connect to DB
    log.info("🗄️  Connecting to pgvector database …")
    conn = get_connection()

    # Resume: skip already-indexed
    if resume:
        existing = get_indexed_chunk_ids(conn)
        before   = len(all_rows)
        all_rows = [r for r in all_rows if r["chunk_id"] not in existing]
        log.info(f"   Resume mode: skipping {before - len(all_rows):,} already-indexed chunks")

    if not all_rows:
        log.info("Nothing to index. All chunks already in database.")
        print_stats(conn)
        conn.close()
        return

    log.info(f"\n🚀 Indexing {len(all_rows):,} chunks …")
    start_time = time.time()
    n_indexed  = 0
    DB_BATCH   = 50  # rows per DB upsert

    for db_batch_start in range(0, len(all_rows), DB_BATCH):
        db_batch   = all_rows[db_batch_start : db_batch_start + DB_BATCH]
        texts      = [row["text_for_embedding"] for row in db_batch]

        # Embed this batch (embedder handles internal sub-batching + rate limiting)
        vectors = embed_texts(texts, batch_size=embed_batch_size, show_progress=False)

        upsert_chunks(conn, db_batch, vectors)
        n_indexed += len(db_batch)

        elapsed = time.time() - start_time
        rate    = n_indexed / elapsed if elapsed > 0 else 0
        eta     = (len(all_rows) - n_indexed) / rate if rate > 0 else 0

        log.info(
            f"   [{n_indexed:>6}/{len(all_rows):>6}] "
            f"{n_indexed/len(all_rows)*100:.1f}% | "
            f"{rate:.1f} chunks/s | "
            f"ETA {eta/60:.1f} min"
        )

    elapsed = time.time() - start_time
    log.info(f"\n✅ Indexing complete: {n_indexed:,} chunks in {elapsed/60:.1f} min")
    print_stats(conn)
    conn.close()


def print_stats(conn):
    stats = get_index_stats(conn)
    log.info("\n📊 Index Stats:")
    log.info(f"   Total chunks:   {stats['total']:,}")
    log.info(f"   With embeddings:{stats['with_embedding']:,}")
    log.info(f"   Books indexed:  {stats['books']}")
    log.info(f"   Surahs covered: {stats['surahs']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book",   type=str,  help="Filter by book slug (e.g. almuyassar)")
    parser.add_argument("--resume", action="store_true", help="Skip already-indexed chunks")
    parser.add_argument("--status", action="store_true", help="Show index stats and exit")
    args = parser.parse_args()

    if args.status:
        conn = get_connection()
        print_stats(conn)
        conn.close()
    else:
        run_indexing(book_filter=args.book, resume=args.resume)
