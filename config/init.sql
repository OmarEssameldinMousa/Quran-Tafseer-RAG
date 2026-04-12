-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For BM25-style trigram full-text search

-- ============================================================
-- MAIN CHUNKS TABLE
-- Each row = one tafseer explanation unit (one or grouped ayahs)
-- ============================================================
CREATE TABLE IF NOT EXISTS tafseer_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        TEXT UNIQUE NOT NULL,          -- {book_slug}_{surah}_{ayah_start}_{ayah_end}

    -- Book metadata
    book_slug       TEXT NOT NULL,                 -- e.g. 'almuyassar'
    book_name_ar    TEXT NOT NULL,                 -- e.g. 'التفسير الميسر'
    book_name_en    TEXT,
    book_api_id     INTEGER NOT NULL,              -- API numeric ID (1-8)
    author          TEXT,

    -- Quran position metadata
    surah_number    INTEGER NOT NULL CHECK (surah_number BETWEEN 1 AND 114),
    surah_name_ar   TEXT NOT NULL,
    surah_name_en   TEXT,
    revelation_type TEXT,                          -- 'مكية' or 'مدنية'
    juz             INTEGER CHECK (juz BETWEEN 1 AND 30),
    hizb            INTEGER,
    page_quran      INTEGER,
    ayah_number_start INTEGER NOT NULL,
    ayah_number_end   INTEGER NOT NULL,

    -- Content
    ayah_text       TEXT NOT NULL,                 -- The Quranic verse (Uthmani)
    tafseer_text    TEXT NOT NULL,                 -- The explanation
    text_for_embedding TEXT NOT NULL,              -- Combined field used for embedding

    -- Stats
    word_count      INTEGER,
    char_count      INTEGER,

    -- The embedding vector (1024-dim for multilingual-e5-large)
    embedding       vector(1024),

    created_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================

-- IVFFlat index for approximate nearest-neighbour vector search
-- Lists = sqrt(n_rows). For ~50K rows: sqrt(50000) ≈ 224 → use 256
CREATE INDEX IF NOT EXISTS tafseer_chunks_embedding_idx
    ON tafseer_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 256);

-- Trigram index for Arabic full-text/BM25-style search
CREATE INDEX IF NOT EXISTS tafseer_chunks_tafseer_trgm_idx
    ON tafseer_chunks USING gin (tafseer_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS tafseer_chunks_ayah_trgm_idx
    ON tafseer_chunks USING gin (ayah_text gin_trgm_ops);

-- Standard b-tree indexes for metadata filtering
CREATE INDEX IF NOT EXISTS tafseer_chunks_book_idx    ON tafseer_chunks (book_slug);
CREATE INDEX IF NOT EXISTS tafseer_chunks_surah_idx   ON tafseer_chunks (surah_number);
CREATE INDEX IF NOT EXISTS tafseer_chunks_juz_idx     ON tafseer_chunks (juz);

-- ============================================================
-- EVALUATION DATASET TABLE
-- Stores auto-generated Q&A pairs for RAGAS-style evaluation
-- ============================================================
CREATE TABLE IF NOT EXISTS eval_qa_pairs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question        TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    reference_chunk_ids TEXT[],                    -- chunk_ids that contain the answer
    surah_number    INTEGER,
    difficulty      TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    question_type   TEXT,                          -- 'factual', 'explanatory', 'comparative'
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- RAG QUERY LOG TABLE  
-- Logs every RAG query for monitoring and performance tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS rag_query_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT NOT NULL,
    query_embedding vector(1024),
    retrieved_chunk_ids TEXT[],
    retrieved_scores FLOAT[],
    generated_answer TEXT,
    -- Performance metrics
    embed_latency_ms  FLOAT,
    retrieve_latency_ms FLOAT,
    generate_latency_ms FLOAT,
    total_latency_ms  FLOAT,
    -- Quality metrics (filled post-eval)
    faithfulness_score FLOAT,
    relevancy_score    FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Confirm setup
SELECT 'pgvector schema initialized successfully ✓' AS status;
