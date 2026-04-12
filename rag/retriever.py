"""
rag/retriever.py
================
Hybrid retrieval combining vector (semantic) + trigram (lexical) search.

Why hybrid?
- Vector search captures semantic meaning (great for paraphrased queries)
- Trigram search captures exact Arabic term matches (great for specific
  Quranic terms, proper nouns, and technical tafseer vocabulary)
- Weighted fusion gives the best of both worlds for Arabic

Scoring formula (Reciprocal Rank Fusion):
    final_score = alpha * vec_score + (1 - alpha) * trgm_score

Usage:
    from rag.retriever import retrieve

    results = retrieve("ما معنى الصراط المستقيم؟", top_k=5)
    # results: list of dicts with chunk data + similarity scores
"""

import sys
import logging
import time
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATABASE_URL, TOP_K, HYBRID_ALPHA
from indexing.embedder import embed_query

log = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def retrieve(
    query: str,
    top_k: int = TOP_K,
    alpha: float = HYBRID_ALPHA,
    book_filter: Optional[str] = None,
    surah_filter: Optional[int] = None,
    juz_filter: Optional[int] = None,
    return_latencies: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """
    Hybrid retrieval: vector + trigram search with RRF fusion.

    Args:
        query:           Arabic query string
        top_k:           Number of results to return
        alpha:           Weight for vector score (1.0=pure vector, 0.0=pure trigram)
        book_filter:     Restrict to one book slug (e.g. 'almuyassar')
        surah_filter:    Restrict to one surah number
        juz_filter:      Restrict to one juz
        return_latencies: If True, also return timing breakdown dict

    Returns:
        List of chunk dicts with added fields: vec_score, trgm_score, final_score
        (If return_latencies=True: tuple of (results, latencies))
    """
    latencies = {}
    t0 = time.time()

    # ── Step 1: Embed query ────────────────────────────────────────────────────
    t_embed = time.time()
    query_vec = embed_query(query)
    latencies["embed_ms"] = (time.time() - t_embed) * 1000

    # ── Step 2: Build WHERE clause for optional filters ────────────────────────
    where_parts = []
    params      = {}
    if book_filter:
        where_parts.append("book_slug = %(book_filter)s")
        params["book_filter"] = book_filter
    if surah_filter:
        where_parts.append("surah_number = %(surah_filter)s")
        params["surah_filter"] = surah_filter
    if juz_filter:
        where_parts.append("juz = %(juz_filter)s")
        params["juz_filter"] = juz_filter

    where_clause_trgm = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    vec_where_parts = where_parts + ["embedding IS NOT NULL"]
    where_clause_vec = "WHERE " + " AND ".join(vec_where_parts)

    # ── Step 3: Run hybrid search ──────────────────────────────────────────────
    t_retrieve = time.time()

    # We fetch top_k * 4 from each sub-search to ensure good fusion coverage
    candidates = top_k * 4
    vec_str    = str(query_vec)

    sql = f"""
    WITH
    -- Vector search: cosine similarity
    vec_search AS (
        SELECT
            chunk_id,
            1 - (embedding <=> %(query_vec)s::vector) AS vec_score,
            ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_vec)s::vector) AS vec_rank
        FROM tafseer_chunks
        {where_clause_vec}
        ORDER BY embedding <=> %(query_vec)s::vector
        LIMIT %(candidates)s
    ),
    -- Trigram search: word similarity over tafseer + ayah text
    trgm_search AS (
        SELECT
            chunk_id,
            GREATEST(
                similarity(tafseer_text, %(query)s),
                similarity(ayah_text, %(query)s)
            ) AS trgm_score,
            ROW_NUMBER() OVER (
                ORDER BY GREATEST(
                    similarity(tafseer_text, %(query)s),
                    similarity(ayah_text, %(query)s)
                ) DESC
            ) AS trgm_rank
        FROM tafseer_chunks
        {where_clause_trgm}
        ORDER BY trgm_score DESC
        LIMIT %(candidates)s
    ),
    -- Fuse: RRF-style weighted fusion over union of candidates
    fused AS (
        SELECT
            COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
            COALESCE(v.vec_score,  0.0) AS vec_score,
            COALESCE(t.trgm_score, 0.0) AS trgm_score,
            %(alpha)s * COALESCE(v.vec_score, 0.0)
            + (1 - %(alpha)s) * COALESCE(t.trgm_score, 0.0) AS final_score
        FROM vec_search  v
        FULL OUTER JOIN trgm_search t USING (chunk_id)
    )
    -- Join back to full chunk data
    SELECT
        tc.*,
        f.vec_score,
        f.trgm_score,
        f.final_score
    FROM fused f
    JOIN tafseer_chunks tc USING (chunk_id)
    ORDER BY f.final_score DESC
    LIMIT %(top_k)s
    """

    params.update({
        "query_vec":  vec_str,
        "query":      query,
        "candidates": candidates,
        "top_k":      top_k,
        "alpha":      alpha,
    })

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    latencies["retrieve_ms"] = (time.time() - t_retrieve) * 1000
    latencies["total_ms"]    = (time.time() - t0) * 1000

    # Convert RealDictRow → plain dicts, drop embedding bytes (too large to pass around)
    results = []
    for row in rows:
        d = dict(row)
        d.pop("embedding", None)          # don't carry 1024-dim vector in results
        d["vec_score"]   = float(d.get("vec_score",   0))
        d["trgm_score"]  = float(d.get("trgm_score",  0))
        d["final_score"] = float(d.get("final_score", 0))
        results.append(d)

    if return_latencies:
        return results, latencies
    return results


def retrieve_with_query_embedding(
    query: str,
    top_k: int = TOP_K,
    **kwargs,
) -> tuple[list[dict], list[float], dict]:
    """
    Like retrieve(), but also returns the raw query embedding vector
    (needed by the Streamlit app to visualize it).
    Returns: (results, query_vector, latencies)
    """
    t0 = time.time()
    query_vec = embed_query(query)
    embed_ms  = (time.time() - t0) * 1000

    results, latencies = retrieve(query, top_k=top_k, return_latencies=True, **kwargs)
    latencies["embed_ms"] = embed_ms  # override with our measurement

    return results, query_vec, latencies
