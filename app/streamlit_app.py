"""
app/streamlit_app.py
====================
Interactive RAG Visualizer — step-by-step visualization of the full
Quran Tafseer RAG pipeline.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import time
import math
import logging
from pathlib import Path
import streamlit as st
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import EMBEDDING_DIM, TOP_K, HYBRID_ALPHA, LLM_MODEL

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="تفسير القرآن — RAG Visualizer",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── RTL + Arabic font CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Tajawal:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Tajawal', sans-serif;
}
.arabic-text {
    font-family: 'Amiri', serif;
    font-size: 1.2em;
    direction: rtl;
    text-align: right;
    line-height: 2;
    background: #fafafa;
    border-right: 4px solid #1b7a4a;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 8px 0;
}
.step-header {
    background: linear-gradient(135deg, #1b4f72, #2e86ab);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 16px 0 8px 0;
    font-weight: 600;
    font-size: 1.1em;
}
.metric-card {
    background: #f0f7f4;
    border: 1px solid #b2dfdb;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.chunk-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 14px;
    margin: 8px 0;
    background: white;
}
.chunk-card.high-score { border-left: 4px solid #27ae60; }
.chunk-card.mid-score  { border-left: 4px solid #f39c12; }
.chunk-card.low-score  { border-left: 4px solid #e74c3c; }
.score-badge {
    display: inline-block;
    background: #2e86ab;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 700;
}
.generated-answer {
    font-family: 'Amiri', serif;
    font-size: 1.15em;
    direction: rtl;
    text-align: right;
    line-height: 2.2;
    background: linear-gradient(135deg, #f0f7f4, #e8f5e9);
    border: 1px solid #a5d6a7;
    border-radius: 8px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ إعدادات RAG")
    st.markdown("---")

    top_k = st.slider("عدد المقاطع المسترجعة (Top-K)", 1, 10, TOP_K)
    alpha = st.slider(
        "نسبة البحث الدلالي (α)",
        0.0, 1.0, HYBRID_ALPHA, 0.05,
        help="1.0 = بحث دلالي بحت | 0.0 = بحث نصي بحت"
    )

    st.markdown("---")
    st.markdown("**فلترة اختيارية:**")
    book_options = {
        "الكل": None,
        "التفسير الميسر": "almuyassar",
        "تفسير الجلالين": "aljalalayn",
        "تفسير السعدي": "alsaadi",
        "تفسير ابن كثير": "ibnkathir",
        "تفسير الوسيط": "alwasit",
        "تفسير البغوي": "albaghawi",
        "تفسير القرطبي": "alqurtubi",
        "تفسير الطبري": "altabari",
    }
    selected_book_label = st.selectbox("كتاب التفسير", list(book_options.keys()))
    book_filter         = book_options[selected_book_label]

    surah_filter = st.number_input("رقم السورة (0 = الكل)", 0, 114, 0)
    surah_filter = int(surah_filter) if surah_filter > 0 else None

    st.markdown("---")
    st.markdown(f"**النموذج اللغوي:** `{LLM_MODEL}`")
    st.markdown("**نموذج التضمين:** `multilingual-e5-large`")


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📖 تفسير القرآن الكريم — RAG Visualizer")
st.markdown("نظام استرجاع وتوليد تفاعلي لتفسير القرآن الكريم مع تصور كل خطوة")
st.markdown("---")


# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown('<div class="step-header">🔍 الخطوة 1: أدخل سؤالك</div>', unsafe_allow_html=True)

query = st.text_area(
    "اكتب سؤالك بالعربية:",
    placeholder="مثال: ما معنى الصراط المستقيم في سورة الفاتحة؟",
    height=80,
    key="query_input",
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 3])
with col_btn1:
    run_rag = st.button("🚀 تشغيل RAG", type="primary", use_container_width=True)
with col_btn2:
    clear   = st.button("🗑️ مسح", use_container_width=True)

if clear:
    st.rerun()


# ── Pipeline ───────────────────────────────────────────────────────────────────
if run_rag and query.strip():

    # ── STEP 2: Embedding ──────────────────────────────────────────────────────
    st.markdown('<div class="step-header">🔢 الخطوة 2: تضمين الاستعلام (Query Embedding)</div>',
                unsafe_allow_html=True)

    with st.spinner("جاري تضمين الاستعلام …"):
        try:
            from indexing.embedder import embed_query
            t0        = time.time()
            query_vec = embed_query(query)
            embed_ms  = (time.time() - t0) * 1000
        except Exception as e:
            st.error(f"❌ خطأ في التضمين: {e}")
            st.stop()

    # Visualize embedding
    vec_arr = np.array(query_vec)
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("أبعاد المتجه", f"{len(query_vec):,}")
    with col_e2:
        st.metric("زمن التضمين", f"{embed_ms:.0f} ms")
    with col_e3:
        st.metric("الحجم الأقصي", f"{vec_arr.max():.4f}")

    with st.expander("👁️ عرض أول 30 قيمة من المتجه"):
        st.bar_chart(vec_arr[:30])

    with st.expander("📊 توزيع قيم المتجه"):
        import pandas as pd
        hist_data = pd.DataFrame({"القيمة": vec_arr})
        st.area_chart(hist_data)

    # ── STEP 3: Retrieval ──────────────────────────────────────────────────────
    st.markdown('<div class="step-header">🔎 الخطوة 3: استرجاع المقاطع ذات الصلة (Hybrid Retrieval)</div>',
                unsafe_allow_html=True)

    with st.spinner("جاري البحث في قاعدة البيانات …"):
        try:
            from rag.retriever import retrieve
            chunks, latencies = retrieve(
                query,
                top_k=top_k,
                alpha=alpha,
                book_filter=book_filter if book_filter else None,
                surah_filter=surah_filter,
                return_latencies=True,
            )
        except Exception as e:
            st.error(f"❌ خطأ في الاسترجاع: {e}")
            st.stop()

    # Retrieval metrics
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("مقاطع مسترجعة", len(chunks))
    with col_r2:
        st.metric("زمن الاسترجاع", f"{latencies.get('retrieve_ms', 0):.0f} ms")
    with col_r3:
        books_found = len({c["book_slug"] for c in chunks})
        st.metric("كتب مختلفة", books_found)
    with col_r4:
        avg_score = np.mean([c["final_score"] for c in chunks]) if chunks else 0
        st.metric("متوسط الصلة", f"{avg_score:.3f}")

    # Display chunks
    if chunks:
        st.markdown("**المقاطع المسترجعة (مرتبة حسب الصلة):**")

        for i, chunk in enumerate(chunks, 1):
            score = chunk["final_score"]
            score_class = "high-score" if score > 0.7 else ("mid-score" if score > 0.4 else "low-score")

            with st.expander(
                f"[{i}] {chunk['book_name_ar']} | سورة {chunk['surah_name_ar']} ({chunk['surah_number']}) | آية {chunk['ayah_number_start']}  —  صلة: {score:.3f}",
                expanded=(i <= 2)
            ):
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("الدرجة الكلية",  f"{chunk['final_score']:.4f}")
                with col_s2:
                    st.metric("دلالي (Vector)", f"{chunk['vec_score']:.4f}")
                with col_s3:
                    st.metric("نصي (Trigram)",  f"{chunk['trgm_score']:.4f}")

                st.markdown("**نص الآية:**")
                st.markdown(f'<div class="arabic-text">{chunk["ayah_text"]}</div>',
                            unsafe_allow_html=True)
                st.markdown("**التفسير:**")
                st.markdown(f'<div class="arabic-text">{chunk["tafseer_text"][:600]}{"…" if len(chunk["tafseer_text"]) > 600 else ""}</div>',
                            unsafe_allow_html=True)

        # Similarity heatmap between query and each chunk
        with st.expander("📈 مخطط الصلة الدلالية"):
            scores = [c["final_score"] for c in chunks]
            labels = [f"[{i+1}] {c['surah_name_ar']}/{c['ayah_number_start']}" for i, c in enumerate(chunks)]
            import pandas as pd
            df_scores = pd.DataFrame({"المقطع": labels, "درجة الصلة": scores})
            st.bar_chart(df_scores.set_index("المقطع"))
    else:
        st.warning("⚠️ لم يتم العثور على مقاطع ذات صلة. تأكد من أن قاعدة البيانات تحتوي على بيانات.")
        st.stop()

    # ── STEP 4: Augmentation ──────────────────────────────────────────────────
    st.markdown('<div class="step-header">🧩 الخطوة 4: بناء الطلب الموسّع (Augmentation)</div>',
                unsafe_allow_html=True)

    from rag.generator import build_prompt
    sys_prompt, user_msg = build_prompt(query, chunks)

    with st.expander("👁️ عرض الطلب الكامل المرسل للنموذج"):
        st.markdown("**رسالة النظام (System Prompt):**")
        st.code(sys_prompt, language=None)
        st.markdown("**رسالة المستخدم (User Message):**")
        st.text_area("", user_msg, height=300, disabled=True)

    token_estimate = len(user_msg.split()) * 1.5  # rough estimate
    st.info(f"📏 الطلب يحتوي على ~{int(token_estimate):,} رمز مقدراً | {len(chunks)} مقطع تفسيري")

    # ── STEP 5: Generation ────────────────────────────────────────────────────
    st.markdown('<div class="step-header">🤖 الخطوة 5: التوليد (Generation — Streaming)</div>',
                unsafe_allow_html=True)

    answer_placeholder = st.empty()
    generated_text     = ""
    gen_start          = time.time()
    token_count        = 0

    try:
        from rag.generator import generate
        stream_gen = generate(query, chunks, stream=True)

        with st.spinner(""):
            for token in stream_gen:
                generated_text += token
                token_count    += 1
                # Update display every 3 tokens for smoother streaming
                if token_count % 3 == 0:
                    answer_placeholder.markdown(
                        f'<div class="generated-answer">{generated_text}▌</div>',
                        unsafe_allow_html=True,
                    )

        # Final display without cursor
        answer_placeholder.markdown(
            f'<div class="generated-answer">{generated_text}</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"❌ خطأ في التوليد: {e}")
        st.stop()

    gen_ms = (time.time() - gen_start) * 1000

    # ── STEP 6: Performance Dashboard ────────────────────────────────────────
    st.markdown('<div class="step-header">📊 الخطوة 6: لوحة الأداء</div>',
                unsafe_allow_html=True)

    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        st.metric("تضمين الاستعلام", f"{embed_ms:.0f} ms")
    with col_p2:
        st.metric("الاسترجاع", f"{latencies.get('retrieve_ms', 0):.0f} ms")
    with col_p3:
        st.metric("التوليد", f"{gen_ms:.0f} ms")
    with col_p4:
        total_ms = embed_ms + latencies.get("retrieve_ms", 0) + gen_ms
        st.metric("الإجمالي", f"{total_ms/1000:.2f} ث")
    with col_p5:
        tps = len(generated_text.split()) / (gen_ms / 1000) if gen_ms > 0 else 0
        st.metric("كلمات/ثانية", f"{tps:.1f}")

    # Faithfulness indicator (heuristic: does the answer cite chunk content?)
    with st.expander("📋 مؤشرات الجودة (تقديرية)"):
        # Heuristic: check overlap between answer words and retrieved tafseer words
        answer_words  = set(generated_text.split())
        chunk_words   = set()
        for c in chunks:
            chunk_words.update(c["tafseer_text"].split())
        overlap_ratio = len(answer_words & chunk_words) / max(len(answer_words), 1)

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.metric("تداخل المفردات (تقديري)", f"{overlap_ratio*100:.1f}%",
                      help="نسبة الكلمات المشتركة بين الإجابة والمقاطع المسترجعة")
        with col_q2:
            word_count = len(generated_text.split())
            st.metric("عدد كلمات الإجابة", word_count)
        with col_q3:
            diversity = len({c["book_slug"] for c in chunks}) / max(top_k, 1)
            st.metric("تنوع المصادر", f"{diversity*100:.0f}%",
                      help="نسبة الكتب المختلفة من إجمالي المقاطع المسترجعة")

        # Pipeline summary bar
        labels = ["التضمين", "الاسترجاع", "التوليد"]
        times  = [embed_ms, latencies.get("retrieve_ms", 0), gen_ms]
        import pandas as pd
        df_times = pd.DataFrame({"المرحلة": labels, "الوقت (ms)": times})
        st.bar_chart(df_times.set_index("المرحلة"))

    st.success("✅ اكتملت عملية RAG بنجاح!")

elif run_rag and not query.strip():
    st.warning("⚠️ يرجى إدخال سؤال أولاً")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>🕌 نظام تفسير القرآن الكريم | مبني باستخدام pgvector + HuggingFace + Groq</small>",
    unsafe_allow_html=True,
)
