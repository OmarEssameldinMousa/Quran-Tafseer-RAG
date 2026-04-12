# 🕌 Arabic Quran Tafseer RAG System

A Retrieval-Augmented Generation system for Quran Tafseer in Arabic, featuring an interactive step-by-step visualization Streamlit app, hybrid vector+trigram search, and a publishable Arabic NLP dataset.

---

## 🏗️ Architecture

```
Query (Arabic)
    │
    ▼
[Step 1] HuggingFace Inference API
         intfloat/multilingual-e5-large
         → 1024-dim query vector
    │
    ▼
[Step 2] pgvector (PostgreSQL)
         Hybrid Search:
           α  × cosine_similarity(query_vec, chunk_vec)   ← semantic
         + (1-α) × trigram_similarity(query, chunk_text)  ← lexical
         → Top-K ranked chunks
    │
    ▼
[Step 3] Prompt Augmentation
         System prompt + retrieved chunks → full Arabic prompt
    │
    ▼
[Step 4] Groq API (llama-3.3-70b-versatile)
         Streaming Arabic generation
    │
    ▼
[Output] Answer + visualized pipeline metrics
```

---

## 📦 Dataset

8 classical Arabic Tafseer books — ~49,000 records:

| # | Book | Author |
|---|------|--------|
| 1 | التفسير الميسر | مجمع الملك فهد |
| 2 | تفسير الجلالين | المحلي والسيوطي |
| 3 | تفسير السعدي | ابن ناصر السعدي |
| 4 | تفسير ابن كثير | إسماعيل بن كثير |
| 5 | تفسير الوسيط | محمد سيد طنطاوي |
| 6 | تفسير البغوي | الحسين بن مسعود البغوي |
| 7 | تفسير القرطبي | محمد بن أحمد القرطبي |
| 8 | تفسير الطبري | محمد بن جرير الطبري |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Free API keys:
  - [HuggingFace](https://huggingface.co/settings/tokens) (for embeddings)
  - [Groq](https://console.groq.com/keys) (for generation)

### 1. Clone & Configure
```bash
git clone <your-repo>
cd quran_tafseer_rag
cp .env.example .env
# Edit .env with your HF_TOKEN and GROQ_API_KEY
```

### 2. Start pgvector Database
```bash
docker compose up -d

# Verify it's running
docker compose ps
docker compose logs pgvector
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Collect Data (fetch all 8 books)
```bash
# Full collection (~2-3 hours for all 8 books due to API rate limits)
python -m data.collector

# Quick test: only Surah Al-Fatiha
python -m data.collector --surah 1

# Only specific books
python -m data.collector --books 1 3
```
Output: `data/processed/master_tafseer_dataset.csv` (~49K rows)

### 5. Index into pgvector
```bash
# Full index (slow due to HF free tier: ~0.8 req/sec)
# Estimated time: 49K chunks / 8 chunks per batch / 0.8 RPS ≈ ~2 hours
python -m indexing.indexer

# Resume after interruption
python -m indexing.indexer --resume

# Check progress
python -m indexing.indexer --status
```

### 6. Generate Evaluation Dataset
```bash
python -m eval.dataset_generator --n 200
```

### 7. Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501

---

## ⚡ Rate Limit Architecture

### HuggingFace (Embedding — free tier)
The embedder uses a **triple-layer protection** to never hit 429:

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| 1 | Token Bucket Limiter | Hard cap at 0.8 req/sec |
| 2 | Adaptive Throttler | Auto-slows after any 429 |
| 3 | Exponential Backoff | Retry with jitter (max 2 min) |

### Groq (Generation — free tier)
- 14,400 requests/day, 70,000 tokens/minute
- Each RAG query uses 1 request (~500-800 tokens)
- Well within free tier for development/research use

---

## 📂 Project Structure

```
quran_tafseer_rag/
├── docker-compose.yml          # pgvector database
├── config/
│   ├── init.sql                # DB schema (auto-runs on first start)
│   └── settings.py             # All configuration
├── data/
│   ├── collector.py            # Async fetch from quran-tafseer API
│   ├── raw/                    # Raw JSON per book (auto-generated)
│   └── processed/              # CSV files (auto-generated)
│       ├── almuyassar.csv
│       ├── ...
│       ├── master_tafseer_dataset.csv
│       └── eval_qa_pairs.csv
├── indexing/
│   ├── embedder.py             # HF API embedding with rate protection
│   └── indexer.py              # pgvector upsert pipeline
├── rag/
│   ├── retriever.py            # Hybrid vector + trigram search
│   └── generator.py            # Groq LLM generation + augmentor
├── eval/
│   └── dataset_generator.py   # Auto-generate Q&A pairs via Groq
├── app/
│   └── streamlit_app.py        # Interactive RAG visualizer
├── .env.example                # Config template
└── requirements.txt
```

---

## 🗄️ Database Management

```bash
# Stop database (data persists in Docker volume)
docker compose stop

# Restart
docker compose start

# init the database and schemas
docker exec -i quran_tafseer_pgvector psql -U tafseer -d quran_rag < config/init.sql

# View logs
docker compose logs -f pgvector

# Connect with psql
docker exec -it quran_tafseer_pgvector psql -U tafseer -d quran_rag

# DANGER: Full reset (deletes all data)
docker compose down -v
```

---

## 🔬 Key Design Decisions

**Why HuggingFace for embeddings?**
The only free provider hosting `intfloat/multilingual-e5-large` exactly. OpenRouter and DeepInfra host it but require payment. HF's `hf-inference` CPU tier is free indefinitely.

**Why Groq for generation?**
Free tier with 14,400 req/day, no card required. Llama 3.3 70B has strong Arabic quality. 500-800 tokens/sec makes streaming feel instant.

**Why hybrid search (vector + trigram)?**
Arabic has rich morphology — the same root appears in many forms. Pure vector search may miss exact term matches for specific Quranic vocabulary. Trigrams catch these while vectors handle semantic paraphrasing.

**Why keep tafseer chunks atomic (one ayah = one chunk)?**
Splitting tafseer mid-explanation destroys context and meaning. Each ayah's explanation is self-contained in classical books.

---

## 📜 License

Dataset sourced from [api.quran-tafseer.com](http://api.quran-tafseer.com) (MIT) and [Tanzil.net](https://tanzil.net).
Code is MIT licensed.
