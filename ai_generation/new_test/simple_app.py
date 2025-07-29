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
# 1. Interface
# ──────────────────────────────────────────────
st.set_page_config(page_title="Mini-pipeline Blog", page_icon="📝", layout="wide")
st.title("📝 Article · 📰 Résumé · 🎨 Image")

topic = st.text_input("Sujet de l'article", placeholder="Ex. : L'IA générative en 2025")
gen_image = st.checkbox("Générer une illustration avec Stable Diffusion", value=True)
#temperature = st.slider("Créativité (temperature GPT-2)",0.2, 0.8, 0.35, 0.05)
btn = st.button("Générer le pipeline")

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
    return ImageGenerator(steps=15)                       # Stable Diffusion Turbo

@st.cache_resource(show_spinner=False)
def get_similarity():
    return SimilarityChecker()                    # MiniLM embedder

gen   = get_article_gen()
summ  = get_summarizer()
imgg  = get_img_gen()
simch = get_similarity()



# ──────────────────────────────────────────────
# 2. Pipeline
# ──────────────────────────────────────────────
if btn and topic.strip():
    # 2.1 Article
    with st.spinner("🧠 Génération de l'article…"):
        article = gen.generate(topic)
    if article.lower().count("sexy") > 5 or article.count("Article:") > 5:
        st.error("⚠️ Article incohérent ou absurde détecté. Veuillez réessayer ou ajuster la température.")
        st.stop()

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
    img_path = None  # Par défaut

    if gen_image:
        prompt = f"{topic}, vector illustration, flat design, clean lines, vibrant colors"
        with st.spinner("🎨 Génération de l'image…"):
            tmp_dir  = tempfile.gettempdir()
            img_path = os.path.join(tmp_dir, f"img_{dt.datetime.now():%H%M%S}.png")
            img_path = imgg.generate(
                prompt,
                negative="text, watermark, lowres, distorted, blurry",
                path=img_path
            )



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

    st.subheader("Filtrage éthique")
    if filt["flagged"]:
        st.error(f"⚠️ Contenu sensible détecté : {', '.join(filt['labels'])}")
        st.json(filt["scores"])
    else:
        st.success("✅ Aucun contenu problématique détecté.")

    if gen_image and img_path:
        st.subheader("Illustration générée")
        st.image(img_path, caption="Stable Diffusion")