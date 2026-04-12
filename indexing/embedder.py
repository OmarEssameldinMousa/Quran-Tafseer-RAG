"""
indexing/embedder.py
====================
Embeds text using intfloat/multilingual-e5-large via HuggingFace
Inference API (free tier).

KEY DESIGN: Triple-layer rate-limit protection so we NEVER hit 429:
  1. Token Bucket Limiter  — hard cap on req/sec (0.8 RPS = 48 RPM)
  2. Adaptive Throttler    — slows down automatically after any 429
  3. Exponential Backoff   — graceful retry with jitter on failures

The HF free tier uses CPU inference for embedding models, so expect
~1-3 sec per request. Our 0.8 RPS budget is conservative and safe.

Usage:
    from indexing.embedder import embed_texts, embed_query

    # Embed a list of passages (for indexing)
    vectors = embed_texts(["passage: نص هنا", "passage: نص آخر"])

    # Embed a single query (for retrieval)
    vector = embed_query("ما معنى الصراط المستقيم؟")
"""

import time
import random
import logging
import requests
import threading
import numpy as np
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    HF_TOKEN, HF_API_URL,
    HF_RATE_LIMIT_RPS, HF_BATCH_SIZE,
    HF_MAX_RETRIES, HF_RETRY_BASE_DELAY, HF_RETRY_MAX_DELAY,
    EMBEDDING_DIM,
)

log = logging.getLogger(__name__)

# ── Token Bucket Rate Limiter (thread-safe) ────────────────────────────────────
class TokenBucketLimiter:
    """
    Thread-safe token bucket limiter.
    Guarantees we never exceed HF_RATE_LIMIT_RPS requests per second.
    Even if called from multiple threads simultaneously, the lock
    ensures sequential access with the enforced inter-request gap.
    """
    def __init__(self, rps: float):
        self._min_interval = 1.0 / rps
        self._last_call    = 0.0
        self._lock         = threading.Lock()

    def wait(self):
        """Block until it's safe to make a new request."""
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_call
            gap     = self._min_interval - elapsed
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.monotonic()


# ── Adaptive Throttler ────────────────────────────────────────────────────────
class AdaptiveThrottler:
    """
    Increases delay after 429 errors, decreases after successes.
    Works on top of the token bucket as a second safety layer.
    """
    def __init__(self):
        self._extra_delay = 0.0
        self._lock        = threading.Lock()

    def on_success(self):
        with self._lock:
            # Gradually recover: reduce extra delay by 10% on each success
            self._extra_delay = max(0.0, self._extra_delay * 0.9)

    def on_rate_limit(self, retry_after: float = 10.0):
        with self._lock:
            # Aggressively back off: set extra delay to retry_after
            self._extra_delay = max(self._extra_delay * 2, retry_after)
            log.warning(f"⚠️  Adaptive throttler: extra delay set to {self._extra_delay:.1f}s")

    def wait(self):
        with self._lock:
            d = self._extra_delay
        if d > 0:
            time.sleep(d)


# ── Singletons ─────────────────────────────────────────────────────────────────
_bucket    = TokenBucketLimiter(rps=HF_RATE_LIMIT_RPS)
_throttler = AdaptiveThrottler()


# ── Core API call ──────────────────────────────────────────────────────────────
def _call_hf_api(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Call HuggingFace Inference API for feature extraction.
    Returns a list of embedding vectors, or None on permanent failure.

    HF returns shape: [n_texts, seq_len, dim] — we mean-pool over seq_len.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs":     texts,
        "parameters": {"normalize": True},  # L2-normalize for cosine similarity
    }

    for attempt in range(HF_MAX_RETRIES):
        # Layer 1: Token bucket — enforces max RPS
        _bucket.wait()
        # Layer 2: Adaptive throttler — slows down after 429s
        _throttler.wait()

        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=120,  # HF CPU inference can be slow
            )

            if response.status_code == 200:
                raw = response.json()
                _throttler.on_success()

                # HF returns [n_texts][seq_len][dim] — mean pool over seq_len
                vectors = []
                for item in raw:
                    arr = np.array(item)
                    if arr.ndim == 2:          # [seq_len, dim] → mean over tokens
                        vec = arr.mean(axis=0)
                    elif arr.ndim == 1:        # [dim] — already pooled
                        vec = arr
                    else:
                        vec = arr.reshape(-1, EMBEDDING_DIM).mean(axis=0)
                    vectors.append(vec.tolist())
                return vectors

            elif response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 30))
                _throttler.on_rate_limit(retry_after)
                jitter = random.uniform(0, 5)
                wait   = min(retry_after + jitter, HF_RETRY_MAX_DELAY)
                log.warning(
                    f"HF 429 — Rate limited (attempt {attempt+1}/{HF_MAX_RETRIES}). "
                    f"Waiting {wait:.1f}s …"
                )
                time.sleep(wait)

            elif response.status_code == 503:
                # Model loading — HF sometimes needs cold-start time
                wait = min(HF_RETRY_BASE_DELAY ** (attempt + 1), HF_RETRY_MAX_DELAY)
                log.info(f"HF 503 — Model loading. Waiting {wait:.0f}s … (attempt {attempt+1})")
                time.sleep(wait)

            else:
                log.error(f"HF API error {response.status_code}: {response.text[:200]}")
                wait = min(HF_RETRY_BASE_DELAY ** attempt, HF_RETRY_MAX_DELAY)
                time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = min(HF_RETRY_BASE_DELAY ** attempt, HF_RETRY_MAX_DELAY)
            log.warning(f"HF timeout (attempt {attempt+1}). Retrying in {wait:.0f}s …")
            time.sleep(wait)

        except requests.exceptions.ConnectionError as e:
            wait = min(HF_RETRY_BASE_DELAY ** attempt * 2, HF_RETRY_MAX_DELAY)
            log.warning(f"HF connection error: {e}. Retrying in {wait:.0f}s …")
            time.sleep(wait)

    log.error(f"HF API: All {HF_MAX_RETRIES} attempts failed for batch of {len(texts)}")
    return None


# ── Public interface ───────────────────────────────────────────────────────────
def embed_texts(
    texts: list[str],
    batch_size: int = HF_BATCH_SIZE,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of passage texts in batches.
    Texts should already have the 'passage: ' prefix.

    Returns list of embedding vectors (1024-dim each).
    Failed batches return zero-vectors (logged as errors).
    """
    all_vectors = []
    n_batches   = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch      = texts[i : i + batch_size]
        batch_num  = i // batch_size + 1

        if show_progress:
            log.info(f"Embedding batch {batch_num}/{n_batches} ({len(batch)} texts) …")

        vectors = _call_hf_api(batch)

        if vectors is None:
            log.error(f"Batch {batch_num} failed — using zero vectors as fallback")
            vectors = [[0.0] * EMBEDDING_DIM] * len(batch)

        all_vectors.extend(vectors)

    return all_vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Automatically adds the 'query: ' prefix required by E5.
    """
    prefixed = f"query: {query}"
    results  = _call_hf_api([prefixed])
    if results:
        return results[0]
    log.error("Query embedding failed — returning zero vector")
    return [0.0] * EMBEDDING_DIM


def embed_texts_with_prefix(raw_texts: list[str], prefix: str = "passage") -> list[list[float]]:
    """
    Embed texts, auto-adding the E5 prefix.
    prefix: 'passage' for documents, 'query' for queries.
    """
    prefixed = [f"{prefix}: {t}" for t in raw_texts]
    return embed_texts(prefixed)


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not HF_TOKEN or HF_TOKEN.startswith("hf_xxx"):
        print("❌ Please set HF_TOKEN in your .env file")
        sys.exit(1)

    print("Testing HuggingFace embedding API …")
    test_texts = [
        "passage: سورة الفاتحة - الآية 1\nالآية: بسم الله الرحمن الرحيم\nالتفسير: أبتدئ قراءة القرآن باسم الله",
        "passage: سورة البقرة - الآية 255\nالآية: الله لا إله إلا هو الحي القيوم\nالتفسير: آية الكرسي",
    ]

    t0      = time.time()
    vectors = embed_texts(test_texts)
    elapsed = time.time() - t0

    print(f"\n✅ Success! Got {len(vectors)} vectors in {elapsed:.2f}s")
    print(f"   Vector dimension: {len(vectors[0])}")
    print(f"   First 5 values:   {vectors[0][:5]}")

    # Test cosine similarity
    v1  = np.array(vectors[0])
    v2  = np.array(vectors[1])
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"   Cosine similarity between test texts: {sim:.4f}")

    q_vec = embed_query("ما معنى بسم الله الرحمن الرحيم؟")
    print(f"\n✅ Query vector dim: {len(q_vec)}")
