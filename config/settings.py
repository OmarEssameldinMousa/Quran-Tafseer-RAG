"""
config/settings.py
Central configuration loaded from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Database ──────────────────────────────────────────────────
POSTGRES_USER     = os.getenv("POSTGRES_USER", "tafseer")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "tafseer_secret")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "quran_rag")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Embedding ─────────────────────────────────────────────────
HF_TOKEN         = os.getenv("HF_TOKEN", "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM", "1024"))

# HuggingFace Inference API endpoint for feature extraction
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"

# Rate-limit safety config for HF free tier
# HF allows ~1 req/sec for embedding on free tier
# We stay at 0.8 req/sec = 48 req/min (well under any limit)
HF_RATE_LIMIT_RPS      = 0.8   # max requests per second
HF_BATCH_SIZE          = 8     # texts per API call (reduces total calls)
HF_MAX_RETRIES         = 8     # exponential backoff retries
HF_RETRY_BASE_DELAY    = 2.0   # seconds (doubles each retry)
HF_RETRY_MAX_DELAY     = 120.0 # cap backoff at 2 minutes

# ── LLM ──────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── RAG ──────────────────────────────────────────────────────
TOP_K          = int(os.getenv("TOP_K", "5"))
HYBRID_ALPHA   = float(os.getenv("HYBRID_ALPHA", "0.7"))  # 1.0=pure vector, 0.0=pure trigram

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_RAW_DIR  = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Tafseer Books ─────────────────────────────────────────────
# Maps API book IDs to human-readable metadata
TAFSEER_BOOKS = {
    1: {
        "slug": "almuyassar",
        "name_ar": "التفسير الميسر",
        "name_en": "Al-Muyassar",
        "author": "مجمع الملك فهد لطباعة المصحف الشريف",
    },
    2: {
        "slug": "aljalalayn",
        "name_ar": "تفسير الجلالين",
        "name_en": "Tafsir Al-Jalalayn",
        "author": "جلال الدين المحلي وجلال الدين السيوطي",
    },
    3: {
        "slug": "alsaadi",
        "name_ar": "تفسير السعدي",
        "name_en": "Tafsir Al-Saadi",
        "author": "عبد الرحمن بن ناصر السعدي",
    },
    4: {
        "slug": "ibnkathir",
        "name_ar": "تفسير ابن كثير",
        "name_en": "Tafsir Ibn Kathir",
        "author": "إسماعيل بن كثير",
    },
    5: {
        "slug": "alwasit",
        "name_ar": "تفسير الوسيط لطنطاوي",
        "name_en": "Tafsir Al-Wasit",
        "author": "محمد سيد طنطاوي",
    },
    6: {
        "slug": "albaghawi",
        "name_ar": "تفسير البغوي",
        "name_en": "Tafsir Al-Baghawi",
        "author": "الحسين بن مسعود البغوي",
    },
    7: {
        "slug": "alqurtubi",
        "name_ar": "تفسير القرطبي",
        "name_en": "Tafsir Al-Qurtubi",
        "author": "محمد بن أحمد القرطبي",
    },
    8: {
        "slug": "altabari",
        "name_ar": "تفسير الطبري",
        "name_en": "Tafsir Al-Tabari",
        "author": "محمد بن جرير الطبري",
    },
}

# Surah metadata: number → (arabic name, english name, ayah count, revelation type)
SURAH_METADATA = {
    1:  ("الفاتحة",       "Al-Fatiha",        7,   "مكية"),
    2:  ("البقرة",        "Al-Baqara",        286,  "مدنية"),
    3:  ("آل عمران",      "Ali 'Imran",       200,  "مدنية"),
    4:  ("النساء",        "An-Nisa",          176,  "مدنية"),
    5:  ("المائدة",       "Al-Ma'ida",        120,  "مدنية"),
    6:  ("الأنعام",       "Al-An'am",         165,  "مكية"),
    7:  ("الأعراف",       "Al-A'raf",         206,  "مكية"),
    8:  ("الأنفال",       "Al-Anfal",          75,  "مدنية"),
    9:  ("التوبة",        "At-Tawba",         129,  "مدنية"),
    10: ("يونس",          "Yunus",            109,  "مكية"),
    11: ("هود",           "Hud",              123,  "مكية"),
    12: ("يوسف",          "Yusuf",            111,  "مكية"),
    13: ("الرعد",         "Ar-Ra'd",           43,  "مدنية"),
    14: ("إبراهيم",       "Ibrahim",           52,  "مكية"),
    15: ("الحجر",         "Al-Hijr",           99,  "مكية"),
    16: ("النحل",         "An-Nahl",          128,  "مكية"),
    17: ("الإسراء",       "Al-Isra",          111,  "مكية"),
    18: ("الكهف",         "Al-Kahf",          110,  "مكية"),
    19: ("مريم",          "Maryam",            98,  "مكية"),
    20: ("طه",            "Ta-Ha",            135,  "مكية"),
    21: ("الأنبياء",      "Al-Anbya",         112,  "مكية"),
    22: ("الحج",          "Al-Hajj",           78,  "مدنية"),
    23: ("المؤمنون",      "Al-Mu'minun",      118,  "مكية"),
    24: ("النور",         "An-Nur",            64,  "مدنية"),
    25: ("الفرقان",       "Al-Furqan",         77,  "مكية"),
    26: ("الشعراء",       "Ash-Shu'ara",      227,  "مكية"),
    27: ("النمل",         "An-Naml",           93,  "مكية"),
    28: ("القصص",         "Al-Qasas",          88,  "مكية"),
    29: ("العنكبوت",      "Al-'Ankabut",       69,  "مكية"),
    30: ("الروم",         "Ar-Rum",            60,  "مكية"),
    31: ("لقمان",         "Luqman",            34,  "مكية"),
    32: ("السجدة",        "As-Sajda",          30,  "مكية"),
    33: ("الأحزاب",       "Al-Ahzab",          73,  "مدنية"),
    34: ("سبأ",           "Saba",              54,  "مكية"),
    35: ("فاطر",          "Fatir",             45,  "مكية"),
    36: ("يس",            "Ya-Sin",            83,  "مكية"),
    37: ("الصافات",       "As-Saffat",        182,  "مكية"),
    38: ("ص",             "Sad",               88,  "مكية"),
    39: ("الزمر",         "Az-Zumar",          75,  "مكية"),
    40: ("غافر",          "Ghafir",            85,  "مكية"),
    41: ("فصلت",          "Fussilat",          54,  "مكية"),
    42: ("الشورى",        "Ash-Shura",         53,  "مكية"),
    43: ("الزخرف",        "Az-Zukhruf",        89,  "مكية"),
    44: ("الدخان",        "Ad-Dukhan",         59,  "مكية"),
    45: ("الجاثية",       "Al-Jathiya",        37,  "مكية"),
    46: ("الأحقاف",       "Al-Ahqaf",          35,  "مكية"),
    47: ("محمد",          "Muhammad",          38,  "مدنية"),
    48: ("الفتح",         "Al-Fath",           29,  "مدنية"),
    49: ("الحجرات",       "Al-Hujurat",        18,  "مدنية"),
    50: ("ق",             "Qaf",               45,  "مكية"),
    51: ("الذاريات",      "Adh-Dhariyat",      60,  "مكية"),
    52: ("الطور",         "At-Tur",            49,  "مكية"),
    53: ("النجم",         "An-Najm",           62,  "مكية"),
    54: ("القمر",         "Al-Qamar",          55,  "مكية"),
    55: ("الرحمن",        "Ar-Rahman",         78,  "مدنية"),
    56: ("الواقعة",       "Al-Waqi'a",         96,  "مكية"),
    57: ("الحديد",        "Al-Hadid",          29,  "مدنية"),
    58: ("المجادلة",      "Al-Mujadila",       22,  "مدنية"),
    59: ("الحشر",         "Al-Hashr",          24,  "مدنية"),
    60: ("الممتحنة",      "Al-Mumtahana",      13,  "مدنية"),
    61: ("الصف",          "As-Saf",            14,  "مدنية"),
    62: ("الجمعة",        "Al-Jumu'a",         11,  "مدنية"),
    63: ("المنافقون",     "Al-Munafiqun",      11,  "مدنية"),
    64: ("التغابن",       "At-Taghabun",       18,  "مدنية"),
    65: ("الطلاق",        "At-Talaq",          12,  "مدنية"),
    66: ("التحريم",       "At-Tahrim",         12,  "مدنية"),
    67: ("الملك",         "Al-Mulk",           30,  "مكية"),
    68: ("القلم",         "Al-Qalam",          52,  "مكية"),
    69: ("الحاقة",        "Al-Haaqqa",         52,  "مكية"),
    70: ("المعارج",       "Al-Ma'arij",        44,  "مكية"),
    71: ("نوح",           "Nuh",               28,  "مكية"),
    72: ("الجن",          "Al-Jinn",           28,  "مكية"),
    73: ("المزمل",        "Al-Muzzammil",      20,  "مكية"),
    74: ("المدثر",        "Al-Muddaththir",    56,  "مكية"),
    75: ("القيامة",       "Al-Qiyama",         40,  "مكية"),
    76: ("الإنسان",       "Al-Insan",          31,  "مدنية"),
    77: ("المرسلات",      "Al-Mursalat",       50,  "مكية"),
    78: ("النبأ",         "An-Naba",           40,  "مكية"),
    79: ("النازعات",      "An-Nazi'at",        46,  "مكية"),
    80: ("عبس",           "'Abasa",            42,  "مكية"),
    81: ("التكوير",       "At-Takwir",         29,  "مكية"),
    82: ("الانفطار",      "Al-Infitar",        19,  "مكية"),
    83: ("المطففين",      "Al-Mutaffifin",     36,  "مكية"),
    84: ("الانشقاق",      "Al-Inshiqaq",       25,  "مكية"),
    85: ("البروج",        "Al-Buruj",          22,  "مكية"),
    86: ("الطارق",        "At-Tariq",          17,  "مكية"),
    87: ("الأعلى",        "Al-A'la",           19,  "مكية"),
    88: ("الغاشية",       "Al-Ghashiya",       26,  "مكية"),
    89: ("الفجر",         "Al-Fajr",           30,  "مكية"),
    90: ("البلد",         "Al-Balad",          20,  "مكية"),
    91: ("الشمس",         "Ash-Shams",         15,  "مكية"),
    92: ("الليل",         "Al-Layl",           21,  "مكية"),
    93: ("الضحى",         "Ad-Duhaa",          11,  "مكية"),
    94: ("الشرح",         "Ash-Sharh",          8,  "مكية"),
    95: ("التين",         "At-Tin",             8,  "مكية"),
    96: ("العلق",         "Al-'Alaq",          19,  "مكية"),
    97: ("القدر",         "Al-Qadr",            5,  "مكية"),
    98: ("البينة",        "Al-Bayyina",         8,  "مدنية"),
    99: ("الزلزلة",       "Az-Zalzala",         8,  "مدنية"),
    100: ("العاديات",     "Al-'Adiyat",        11,  "مكية"),
    101: ("القارعة",      "Al-Qari'a",         11,  "مكية"),
    102: ("التكاثر",      "At-Takathur",        8,  "مكية"),
    103: ("العصر",        "Al-'Asr",            3,  "مكية"),
    104: ("الهمزة",       "Al-Humaza",          9,  "مكية"),
    105: ("الفيل",        "Al-Fil",             5,  "مكية"),
    106: ("قريش",         "Quraysh",            4,  "مكية"),
    107: ("الماعون",      "Al-Ma'un",           7,  "مكية"),
    108: ("الكوثر",       "Al-Kawthar",         3,  "مكية"),
    109: ("الكافرون",     "Al-Kafirun",         6,  "مكية"),
    110: ("النصر",        "An-Nasr",            3,  "مدنية"),
    111: ("المسد",        "Al-Masad",           5,  "مكية"),
    112: ("الإخلاص",      "Al-Ikhlas",          4,  "مكية"),
    113: ("الفلق",        "Al-Falaq",           5,  "مكية"),
    114: ("الناس",        "An-Nas",             6,  "مكية"),
}

# Juz boundaries: (surah, ayah) where each juz starts
JUZ_BOUNDARIES = [
    (1,1),(2,142),(2,253),(3,92),(4,24),(4,147),(5,82),(6,111),(7,87),(8,41),
    (9,93),(11,6),(12,53),(15,1),(17,1),(18,75),(21,1),(23,1),(25,21),(27,56),
    (29,46),(33,31),(36,28),(39,32),(41,47),(46,1),(51,31),(58,1),(67,1),(78,1),
]
