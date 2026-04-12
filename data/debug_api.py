"""
debug_api.py  (v2) — Run: python debug_api.py
Prints exact JSON structures so the collector knows what to parse.
"""
import requests, json

S = "=" * 60

# ── 1. alquran.cloud Arabic tafseer editions ──────────────────────────────────
print(S)
print("[1] alquran.cloud — all Arabic tafseer editions:")
try:
    r = requests.get("https://api.alquran.cloud/v1/edition?language=ar&type=tafsir", timeout=15)
    for e in r.json().get("data", []):
        print(f"  {e['identifier']:30s}  {e['englishName']}")
except Exception as ex:
    print(f"  ERROR: {ex}")

# ── 2. Multi-edition in one call (verse text + tafseer) ──────────────────────
print(S)
print("[2] alquran.cloud surah 1 — quran-uthmani + ar.muyassar in ONE call:")
try:
    r = requests.get(
        "https://api.alquran.cloud/v1/surah/1/editions/quran-uthmani,ar.muyassar",
        timeout=15
    )
    d = r.json()
    print(f"  code={d.get('code')}  editions returned={len(d['data'])}")
    for ed in d["data"]:
        ident = ed['edition']['identifier']
        sample = ed['ayahs'][0]['text'][:80]
        print(f"  [{ident}] ayah[0]: {sample}…")
except Exception as ex:
    print(f"  ERROR: {ex}")

# ── 3. Test each candidate tafseer edition ────────────────────────────────────
print(S)
print("[3] Testing candidate Arabic tafseer edition identifiers:")
candidates = [
    "ar.muyassar", "ar.jalalayn", "ar.saddi", "ar.wahidi",
    "ar.ibn-kathir", "ar.baghawi", "ar.tabari", "ar.qurtubi",
    "ar.wasit", "ar.muyassar-g",
]
for ident in candidates:
    try:
        r = requests.get(f"https://api.alquran.cloud/v1/surah/1/{ident}", timeout=10)
        d = r.json()
        if d.get("code") == 200:
            s = d["data"]["ayahs"][0]["text"][:70]
            print(f"  ✅ {ident:30s} '{s}…'")
        else:
            print(f"  ❌ {ident:30s} HTTP {d.get('code')}: {d.get('status','')}")
    except Exception as ex:
        print(f"  ❌ {ident:30s} ERROR: {ex}")

# ── 4. spa5k — exact JSON structure ──────────────────────────────────────────
print(S)
print("[4] spa5k cdn.jsdelivr.net — exact structure for ar-tafsir-muyassar surah 1:")
try:
    r = requests.get(
        "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/ar-tafsir-muyassar/1.json",
        timeout=15
    )
    d = r.json()
    t = type(d).__name__
    print(f"  Top-level type: {t}")
    if isinstance(d, list):
        print(f"  List length: {len(d)}")
        print(f"  Item[0] keys: {list(d[0].keys())}")
        print(f"  Item[0]: {json.dumps(d[0], ensure_ascii=False)}")
        if len(d) > 1:
            print(f"  Item[1]: {json.dumps(d[1], ensure_ascii=False)}")
    elif isinstance(d, dict):
        keys = list(d.keys())
        print(f"  Dict keys (first 5): {keys[:5]}")
        print(f"  Value type: {type(d[keys[0]]).__name__}")
        print(f"  d['{keys[0]}']: {json.dumps(d[keys[0]], ensure_ascii=False)[:200]}")
except Exception as ex:
    print(f"  ERROR: {ex}")

# ── 5. spa5k raw GitHub ───────────────────────────────────────────────────────
print(S)
print("[5] spa5k raw.githubusercontent.com — ar-tafsir-muyassar surah 1:")
try:
    r = requests.get(
        "https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/ar-tafsir-muyassar/1.json",
        timeout=15
    )
    d = r.json()
    t = type(d).__name__
    print(f"  Top-level type: {t},  length={len(d)}")
    if isinstance(d, list) and d:
        print(f"  Item[0]: {json.dumps(d[0], ensure_ascii=False)}")
    elif isinstance(d, dict):
        k = list(d.keys())[0]
        print(f"  d['{k}']: {json.dumps(d[k], ensure_ascii=False)[:200]}")
except Exception as ex:
    print(f"  ERROR: {ex}")

print(S)
print("Done. Share this output to determine correct parsing strategy.")