"""
data/collector.py  (v3 — alquran.cloud primary, spa5k fallback)
================================================================
PRIMARY:  api.alquran.cloud  ← confirmed working from your machine
          One call per surah fetches BOTH verse text + tafseer together.
          No rate limits documented; we stay polite at 2 req/sec.

FALLBACK: spa5k CDN (jsdelivr / GitHub raw)
          Per-surah static JSON files. Quran text fetched separately.

The 8 books map to alquran.cloud tafseer edition identifiers.
Run debug_api.py first to confirm which identifiers are live.

Usage:
    python -m data.collector --surah 1          # test Al-Fatiha
    python -m data.collector                    # all 8 books
    python -m data.collector --books 1 3        # specific books
    python -m data.collector --source spa5k     # force spa5k
"""

import asyncio
import aiohttp
import json
import csv
import time
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import SURAH_METADATA, DATA_RAW_DIR, DATA_PROC_DIR, JUZ_BOUNDARIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Book registry ─────────────────────────────────────────────────────────────
# alquran_edition: the identifier used at api.alquran.cloud/v1/surah/{n}/{identifier}
# spa5k_slug:      the path used at cdn.jsdelivr.net/.../tafsir/{slug}/{surah}.json
# NOTE: Run debug_api.py to confirm which alquran_edition values are valid on your machine.
#       If an edition returns 404, set it to None and spa5k will be used instead.
BOOKS = {
    1: {
        "slug": "almuyassar", "name_ar": "التفسير الميسر",
        "name_en": "Al-Muyassar", "author": "مجمع الملك فهد لطباعة المصحف الشريف",
        "alquran_edition": "ar.muyassar",           # confirmed 200 in debug
        "spa5k_slug":      "ar-tafsir-muyassar",
    },
    2: {
        "slug": "aljalalayn", "name_ar": "تفسير الجلالين",
        "name_en": "Tafsir Al-Jalalayn", "author": "جلال الدين المحلي وجلال الدين السيوطي",
        "alquran_edition": "ar.jalalayn",            # test in debug_api.py
        "spa5k_slug":      "ar-tafseer-tanwir-al-miqbas",
    },
    3: {
        "slug": "alsaadi", "name_ar": "تفسير السعدي",
        "name_en": "Tafsir Al-Saadi", "author": "عبد الرحمن بن ناصر السعدي",
        "alquran_edition": "ar.saddi",               # test in debug_api.py
        "spa5k_slug":      "ar-tafseer-al-saddi",
    },
    4: {
        "slug": "ibnkathir", "name_ar": "تفسير ابن كثير",
        "name_en": "Tafsir Ibn Kathir", "author": "إسماعيل بن كثير",
        "alquran_edition": "ar.ibn-kathir",          # test in debug_api.py
        "spa5k_slug":      "ar-tafsir-ibn-kathir",
    },
    5: {
        "slug": "alwasit", "name_ar": "تفسير الوسيط لطنطاوي",
        "name_en": "Tafsir Al-Wasit", "author": "محمد سيد طنطاوي",
        "alquran_edition": "ar.wasit",               # test in debug_api.py
        "spa5k_slug":      "ar-tafsir-al-wasit",
    },
    6: {
        "slug": "albaghawi", "name_ar": "تفسير البغوي",
        "name_en": "Tafsir Al-Baghawi", "author": "الحسين بن مسعود البغوي",
        "alquran_edition": "ar.baghawi",             # test in debug_api.py
        "spa5k_slug":      "ar-tafsir-al-baghawi",
    },
    7: {
        "slug": "alqurtubi", "name_ar": "تفسير القرطبي",
        "name_en": "Tafsir Al-Qurtubi", "author": "محمد بن أحمد القرطبي",
        "alquran_edition": "ar.qurtubi",             # test in debug_api.py
        "spa5k_slug":      "ar-tafseer-al-qurtubi",
    },
    8: {
        "slug": "altabari", "name_ar": "تفسير الطبري",
        "name_en": "Tafsir Al-Tabari", "author": "محمد بن جرير الطبري",
        "alquran_edition": "ar.tabari",              # test in debug_api.py
        "spa5k_slug":      "ar-tafsir-al-tabari",
    },
}

ALQURAN_BASE = "https://api.alquran.cloud/v1"
SPA5K_CDN    = "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir"
SPA5K_RAW    = "https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir"


# ── HTTP helper ───────────────────────────────────────────────────────────────
async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    label: str = "req",
    retries: int = 4,
) -> Optional[dict | list]:
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                body = (await resp.text())[:120]
                log.warning(f"[{label}] HTTP {resp.status} attempt {attempt+1}: {body}")
        except aiohttp.ClientConnectorError as e:
            log.warning(f"[{label}] Connection error attempt {attempt+1}: {e}")
        except asyncio.TimeoutError:
            log.warning(f"[{label}] Timeout attempt {attempt+1}: {url}")
        except Exception as e:
            log.warning(f"[{label}] {type(e).__name__} attempt {attempt+1}: {e}")
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt)
    log.error(f"[{label}] GAVE UP after {retries} attempts: {url}")
    return None


# ── Juz helper ────────────────────────────────────────────────────────────────
def get_juz(surah: int, ayah: int) -> int:
    juz = 1
    for i, (s, a) in enumerate(JUZ_BOUNDARIES):
        if (surah, ayah) >= (s, a):
            juz = i + 1
    return juz


# ── Record builder ────────────────────────────────────────────────────────────
def make_record(book_id, book, surah_num, ayah_num, ayah_text, tafseer_text):
    meta = SURAH_METADATA[surah_num]
    surah_name_ar, surah_name_en, _, revelation = meta
    text_for_embedding = (
        f"passage: سورة {surah_name_ar} - الآية {ayah_num}\n"
        f"الآية: {ayah_text}\n"
        f"التفسير ({book['name_ar']}): {tafseer_text}"
    )
    return {
        "chunk_id":           f"{book['slug']}_{surah_num}_{ayah_num}_{ayah_num}",
        "book_api_id":        book_id,
        "book_slug":          book["slug"],
        "book_name_ar":       book["name_ar"],
        "book_name_en":       book["name_en"],
        "author":             book["author"],
        "surah_number":       surah_num,
        "surah_name_ar":      surah_name_ar,
        "surah_name_en":      surah_name_en,
        "revelation_type":    revelation,
        "ayah_number_start":  ayah_num,
        "ayah_number_end":    ayah_num,
        "juz":                get_juz(surah_num, ayah_num),
        "ayah_text":          ayah_text,
        "tafseer_text":       tafseer_text,
        "text_for_embedding": text_for_embedding,
        "word_count":         len(tafseer_text.split()),
        "char_count":         len(tafseer_text),
    }


# ── SOURCE A: alquran.cloud — 1 call per surah gets verse + tafseer ───────────
async def fetch_surah_alquran(
    session: aiohttp.ClientSession,
    book_id: int,
    surah_num: int,
) -> list[dict]:
    """
    Fetches both Quranic text + tafseer in a single call using the
    multi-edition endpoint: /surah/{n}/editions/quran-uthmani,{tafseer_id}
    Returns list of record dicts for this surah.
    """
    book      = BOOKS[book_id]
    edition   = book["alquran_edition"]
    url       = f"{ALQURAN_BASE}/surah/{surah_num}/editions/quran-uthmani,{edition}"
    data      = await fetch_json(session, url, label=f"alquran/{book['slug']}/s{surah_num}")

    if not data or data.get("code") != 200:
        log.warning(f"  alquran.cloud returned no data for {edition} surah {surah_num}")
        return []

    editions_data = data["data"]
    # editions_data is a list of 2 items: [quran-uthmani, tafseer_edition]
    if len(editions_data) < 2:
        log.warning(f"  Expected 2 editions, got {len(editions_data)} for surah {surah_num}")
        return []

    # Identify which edition is which by checking the identifier
    uthmani_ayahs  = None
    tafseer_ayahs  = None
    for ed in editions_data:
        ident = ed["edition"]["identifier"]
        if ident == "quran-uthmani":
            uthmani_ayahs = ed["ayahs"]
        else:
            tafseer_ayahs = ed["ayahs"]

    if not uthmani_ayahs or not tafseer_ayahs:
        log.error(f"  Could not identify editions in response for surah {surah_num}")
        return []

    # Zip them together by ayah number
    text_by_ayah    = {a["numberInSurah"]: a["text"] for a in uthmani_ayahs}
    tafseer_by_ayah = {a["numberInSurah"]: a["text"] for a in tafseer_ayahs}

    records = []
    for ayah_num, tafseer_text in tafseer_by_ayah.items():
        tafseer_text = (tafseer_text or "").strip()
        if not tafseer_text:
            continue
        ayah_text = text_by_ayah.get(ayah_num, "")
        records.append(make_record(book_id, book, surah_num, ayah_num, ayah_text, tafseer_text))

    return records


async def fetch_book_alquran(
    session: aiohttp.ClientSession,
    book_id: int,
    surah_filter: Optional[int],
) -> list[dict]:
    book   = BOOKS[book_id]
    surahs = [surah_filter] if surah_filter else list(range(1, 115))
    records, failed = [], []

    log.info(f"  📡 alquran.cloud | edition: {book['alquran_edition']}")

    for surah_num in surahs:
        meta       = SURAH_METADATA[surah_num]
        ayah_count = meta[2]
        surah_recs = await fetch_surah_alquran(session, book_id, surah_num)

        if surah_recs:
            records.extend(surah_recs)
            log.info(f"    Surah {surah_num:3d}/{meta[0]:<12} → {len(surah_recs)}/{ayah_count} ayahs ✓")
        else:
            failed.append(surah_num)
            log.warning(f"    Surah {surah_num:3d}/{meta[0]:<12} → 0 ayahs  ⚠️")

        await asyncio.sleep(0.5)   # 2 req/sec max — polite but fast

    if failed:
        log.warning(f"  ⚠️  Failed surahs: {failed[:20]}")
    return records


# ── SOURCE B: spa5k CDN — fallback ────────────────────────────────────────────
_quran_text_cache: dict[int, dict[int, str]] = {}

async def get_uthmani_texts(session, surah_num):
    if surah_num in _quran_text_cache:
        return _quran_text_cache[surah_num]
    url  = f"{ALQURAN_BASE}/surah/{surah_num}/quran-uthmani"
    data = await fetch_json(session, url, label=f"uthmani/s{surah_num}")
    out  = {}
    if data and data.get("code") == 200:
        for a in data["data"]["ayahs"]:
            out[a["numberInSurah"]] = a["text"]
    _quran_text_cache[surah_num] = out
    return out


async def fetch_surah_spa5k(session, book_id, surah_num, prefer_github=False):
    book = BOOKS[book_id]
    slug = book["spa5k_slug"]
    urls = (
        [f"{SPA5K_RAW}/{slug}/{surah_num}.json", f"{SPA5K_CDN}/{slug}/{surah_num}.json"]
        if prefer_github else
        [f"{SPA5K_CDN}/{slug}/{surah_num}.json", f"{SPA5K_RAW}/{slug}/{surah_num}.json"]
    )
    data = None
    for url in urls:
        data = await fetch_json(session, url, label=f"spa5k/{slug}/s{surah_num}")
        if data is not None:
            break

    if not data:
        return []

    # spa5k structure: list of {ayah_number: int, text: str}
    # OR dict {surah_number: [...ayahs...]} — handle both
    ayah_items = []
    if isinstance(data, list):
        ayah_items = data
    elif isinstance(data, dict):
        # Could be keyed by ayah number string or wrapped in surah key
        for v in data.values():
            if isinstance(v, list):
                ayah_items = v
                break
            elif isinstance(v, dict):
                ayah_items.append(v)

    if not ayah_items:
        log.warning(f"  spa5k returned unrecognised structure for {slug}/s{surah_num}: {str(data)[:100]}")
        return []

    texts   = await get_uthmani_texts(session, surah_num)
    records = []
    for item in ayah_items:
        ayah_num     = item.get("ayah_number") or item.get("ayahNumber") or item.get("number")
        tafseer_text = (item.get("text") or item.get("tafseer") or "").strip()
        if not ayah_num or not tafseer_text:
            continue
        ayah_num  = int(ayah_num)
        ayah_text = texts.get(ayah_num, "")
        records.append(make_record(book_id, book, surah_num, ayah_num, ayah_text, tafseer_text))
    return records


async def fetch_book_spa5k(session, book_id, surah_filter, prefer_github=False):
    book   = BOOKS[book_id]
    surahs = [surah_filter] if surah_filter else list(range(1, 115))
    records, failed = [], []

    log.info(f"  📡 spa5k {'GitHub' if prefer_github else 'jsDelivr'} | slug: {book['spa5k_slug']}")

    for surah_num in surahs:
        meta       = SURAH_METADATA[surah_num]
        ayah_count = meta[2]
        surah_recs = await fetch_surah_spa5k(session, book_id, surah_num, prefer_github)

        if surah_recs:
            records.extend(surah_recs)
            log.info(f"    Surah {surah_num:3d}/{meta[0]:<12} → {len(surah_recs)}/{ayah_count} ayahs ✓")
        else:
            failed.append(surah_num)
            log.warning(f"    Surah {surah_num:3d}/{meta[0]:<12} → 0 ayahs  ⚠️")
        await asyncio.sleep(0.1)

    if failed:
        log.warning(f"  ⚠️  {len(failed)} surahs failed in spa5k: {failed[:20]}")
    return records


# ── Auto-probe ────────────────────────────────────────────────────────────────
async def probe_sources(session) -> str:
    log.info("🔍 Probing sources …")

    # Test alquran.cloud with ar.muyassar on surah 1
    d = await fetch_json(
        session,
        f"{ALQURAN_BASE}/surah/1/editions/quran-uthmani,ar.muyassar",
        label="probe-alquran", retries=2
    )
    if d and d.get("code") == 200 and len(d.get("data", [])) >= 2:
        log.info("  ✅ alquran.cloud is reachable and returning data")
        return "alquran"

    d = await fetch_json(
        session, f"{SPA5K_CDN}/ar-tafsir-muyassar/1.json",
        label="probe-spa5k-cdn", retries=2
    )
    if d is not None:
        log.info("  ✅ spa5k jsDelivr is reachable")
        return "spa5k"

    d = await fetch_json(
        session, f"{SPA5K_RAW}/ar-tafsir-muyassar/1.json",
        label="probe-spa5k-github", retries=2
    )
    if d is not None:
        log.info("  ✅ spa5k GitHub raw is reachable")
        return "spa5k-github"

    log.error("❌ ALL sources unreachable. Check internet. Run debug_api.py.")
    return "none"


# ── Save ──────────────────────────────────────────────────────────────────────
def save_json(book_id, records):
    p = DATA_RAW_DIR / f"{BOOKS[book_id]['slug']}_raw.json"
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  💾 {p.name}  ({p.stat().st_size//1024} KB, {len(records)} records)")

def save_csv(book_id, records):
    if not records:
        log.warning(f"  ⚠️  No records for book {book_id} — CSV skipped")
        return
    p = DATA_PROC_DIR / f"{BOOKS[book_id]['slug']}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    log.info(f"  📄 {p.name}  ({len(records):,} rows)")

def save_master(all_records):
    if not all_records:
        log.error("❌ Zero total records — master CSV NOT written")
        return
    p = DATA_PROC_DIR / "master_tafseer_dataset.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader(); w.writerows(all_records)
    mb = p.stat().st_size / 1_048_576
    log.info(f"\n🗂️  MASTER: {p.name}  ({len(all_records):,} rows, {mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
async def run(book_ids=None, surah_filter=None, source="auto"):
    book_ids    = book_ids or list(BOOKS.keys())
    all_records = []
    t0          = time.time()

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=8),
        headers={"User-Agent": "QuranTafseerRAG/3.0 (academic research)"},
    ) as session:

        if source == "auto":
            source = await probe_sources(session)
            if source == "none":
                return []
        log.info(f"\n📡 Source: {source}\n")

        for book_id in book_ids:
            log.info(f"📖 Book {book_id}: {BOOKS[book_id]['name_ar']}")

            if source == "alquran":
                records = await fetch_book_alquran(session, book_id, surah_filter)
                # If this specific edition not available, try spa5k fallback
                if not records:
                    log.warning(f"  alquran.cloud returned 0 records — falling back to spa5k …")
                    records = await fetch_book_spa5k(session, book_id, surah_filter)
            elif source == "spa5k":
                records = await fetch_book_spa5k(session, book_id, surah_filter)
            else:  # spa5k-github
                records = await fetch_book_spa5k(session, book_id, surah_filter, prefer_github=True)

            log.info(f"  ✅ Total: {len(records):,} records for {BOOKS[book_id]['name_ar']}")
            save_json(book_id, records)
            save_csv(book_id, records)
            all_records.extend(records)
            if book_id != book_ids[-1]:
                await asyncio.sleep(1)

    save_master(all_records)
    log.info(f"\n🎉 Done in {time.time()-t0:.1f}s | {len(all_records):,} total records")
    return all_records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--books",  nargs="+", type=int)
    p.add_argument("--surah",  type=int)
    p.add_argument("--source", default="auto",
                   choices=["auto", "alquran", "spa5k", "spa5k-github"])
    args = p.parse_args()
    asyncio.run(run(book_ids=args.books, surah_filter=args.surah, source=args.source))