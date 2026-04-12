"""
rag/augmentor.py & generator.py — combined
==========================================
Augmentor: builds the prompt from retrieved chunks.
Generator: calls Groq LLM (llama-3.3-70b-versatile) with streaming.
"""

import sys
import logging
import time
from pathlib import Path
from typing import Generator

from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE
)

log = logging.getLogger(__name__)

# ── Augmentor ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """أنت عالم إسلامي متخصص في علوم القرآن الكريم وتفسيره.
مهمتك هي الإجابة على الأسئلة المتعلقة بالقرآن الكريم وتفسيره بناءً على النصوص التفسيرية المقدمة لك.

التعليمات:
- أجب بالعربية الفصحى الواضحة
- استند دائماً إلى نصوص التفسير المقدمة
- أذكر اسم التفسير الذي استندت إليه عند الإجابة
- إذا كانت المعلومات غير كافية في النصوص المقدمة، قل ذلك صراحةً
- لا تخترع معلومات غير موجودة في السياق
- كن دقيقاً وموضوعياً في إجاباتك"""


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build the augmented prompt from retrieved chunks.

    Returns:
        (system_prompt, user_message)  — to be passed to the LLM separately
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[مصدر {i}] {chunk['book_name_ar']} | "
            f"سورة {chunk['surah_name_ar']} ({chunk['surah_number']}) | "
            f"الآية {chunk['ayah_number_start']}\n"
            f"نص الآية: {chunk['ayah_text']}\n"
            f"التفسير: {chunk['tafseer_text']}\n"
        )

    context_block = "\n---\n".join(context_parts)

    user_message = f"""السؤال: {query}

---
النصوص التفسيرية ذات الصلة:

{context_block}
---

بناءً على النصوص التفسيرية المقدمة أعلاه، أجب على السؤال بشكل شامل ومفصل."""

    return SYSTEM_PROMPT, user_message


# ── Generator ─────────────────────────────────────────────────────────────────
def create_groq_client() -> Groq:
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_xxx"):
        raise ValueError("GROQ_API_KEY not set. Check your .env file.")
    return Groq(api_key=GROQ_API_KEY)


def generate(
    query: str,
    chunks: list[dict],
    stream: bool = True,
) -> Generator[str, None, None] | tuple[str, dict]:
    """
    Generate an answer from the query + retrieved chunks.

    Args:
        query:  The user's Arabic question
        chunks: Retrieved tafseer chunks from retriever
        stream: If True, yields token-by-token (for Streamlit streaming UI)
                If False, returns (full_text, metrics)

    Streaming usage:
        for token in generate(query, chunks, stream=True):
            print(token, end="", flush=True)

    Non-streaming usage:
        text, metrics = generate(query, chunks, stream=False)
    """
    client = create_groq_client()
    system_prompt, user_message = build_prompt(query, chunks)

    t0 = time.time()

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        stream=stream,
    )

    if stream:
        def token_generator():
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return token_generator()

    else:
        text     = completion.choices[0].message.content
        elapsed  = (time.time() - t0) * 1000
        metrics  = {
            "generate_ms":    elapsed,
            "input_tokens":   completion.usage.prompt_tokens,
            "output_tokens":  completion.usage.completion_tokens,
            "total_tokens":   completion.usage.total_tokens,
            "model":          LLM_MODEL,
        }
        return text, metrics


def generate_full(query: str, chunks: list[dict]) -> tuple[str, dict]:
    """Convenience wrapper for non-streaming generation."""
    return generate(query, chunks, stream=False)
