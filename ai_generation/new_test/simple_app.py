"""
simple_app.py
Mini-pipeline Streamlit : Article ➜ Résumé ➜ Similarité ➜ Filtrage ➜ Image

Dépendances (CPU) :
    pip install streamlit torch transformers diffusers sentence-transformers
"""

import os, tempfile, datetime as dt
import streamlit as st

from article_generator import ArticleGenerator         # distilgpt2
from summarization      import TextSummarizer          # distilbart-xsum
from similarity         import SimilarityChecker       # ton module existant
from ethical_filter_v2  import ethical_filter          # ou ethical_filter
from image_gen          import ImageGenerator          # SD-Turbo

# ──────────────────────────────────────────────
# 0. Mise en cache des modèles (chargés 1 fois)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Chargement article_generator…")
def get_article_gen():
    return ArticleGenerator()                     # modèle + tokenizer

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return TextSummarizer()

@st.cache_resource(show_spinner=False)
def get_img_gen():
    return ImageGenerator()                       # Stable Diffusion Turbo

@st.cache_resource(show_spinner=False)
def get_similarity():
    return SimilarityChecker()                    # MiniLM embedder

gen   = get_article_gen()
summ  = get_summarizer()
imgg  = get_img_gen()
simch = get_similarity()

# ──────────────────────────────────────────────
# 1. Interface
# ──────────────────────────────────────────────
st.set_page_config(page_title="Mini-pipeline Blog", page_icon="📝", layout="wide")
st.title("📝 Article · 📰 Résumé · 🎨 Image")

topic = st.text_input("Sujet de l'article", placeholder="Ex. : L'IA générative en 2025")
temperature = st.slider("Créativité (temperature GPT-2)",0.2, 0.8, 0.35, 0.05)
btn = st.button("Générer le pipeline")

# ──────────────────────────────────────────────
# 2. Pipeline
# ──────────────────────────────────────────────
if btn and topic.strip():
    # 2.1 Article
    with st.spinner("🧠 Génération de l'article…"):
        article = gen.generate(topic, temperature=temperature)

    # 2.2 Résumé
    with st.spinner("📚 Résumé…"):
        summary = summ.summarize(article)

    # 2.3 Similarité article ↔ résumé
    with st.spinner("📏 Similarité article / résumé…"):
        sim_score = simch.compare(article, summary)   # 0-1

    # 2.4 Filtrage éthique
    with st.spinner("🔍 Filtrage éthique…"):
        filt = ethical_filter(summary)

    # 2.5 Image
    with st.spinner("🎨 Génération de l'image…"):
        tmp_dir  = tempfile.gettempdir()
        img_path = os.path.join(tmp_dir, f"img_{dt.datetime.now():%H%M%S}.png")
        img_path = imgg.generate(topic, path=img_path)

    # ──────────────────────────────────────────
    # 3. Affichage
    # ──────────────────────────────────────────
    st.subheader("Article complet")
    st.write(article)

    st.subheader("Résumé")
    st.write(summary)

    st.subheader("📏 Similarité résumé / article")
    bar_color = "#2ecc71" if sim_score >= 0.5 else "#e67e22" if sim_score >= 0.3 else "#e74c3c"
    st.markdown(
        f"""
        <div style="background:{bar_color};padding:6px;border-radius:6px;
                    color:white;width:{sim_score*100:.1f}%">
        {sim_score:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )
    max_try = 3
    for attempt in range(1, max_try + 1):
        with st.spinner(f"🧠 Génération de l'article… (essai {attempt})"):
            article = gen.generate(topic, temperature=temperature)

        sim_score = simch.compare(article, topic)      # 0-1
        if sim_score >= 0.15:
            break
        st.warning("Le texte semble peu corrélé au sujet ; nouvelle tentative…")

    else:  # boucle terminée sans break
        st.error("Impossible d'obtenir un article suffisamment lié au sujet.")
        st.stop()

    st.subheader("Filtrage éthique")
    if filt["flagged"]:
        st.error(f"⚠️ Contenu sensible détecté : {', '.join(filt['labels'])}")
        st.json(filt["scores"])
    else:
        st.success("✅ Aucun contenu problématique détecté.")

    st.subheader("Illustration générée")
    st.image(img_path, caption="Stable Diffusion Turbo")