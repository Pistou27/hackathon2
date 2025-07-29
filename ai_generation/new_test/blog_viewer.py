import streamlit as st
from pathlib import Path
import json

st.set_page_config(page_title="📚 Mon Blog Génératif", layout="wide")
st.title("📰 Blog généré automatiquement")

# Chemin racine contenant les sous-dossiers d'articles
output_dir = Path("outputs")
article_dirs = sorted(output_dir.glob("*/"), reverse=True)

if not article_dirs:
    st.info("Aucun article généré pour le moment.")
else:
    for article_dir in article_dirs:
        md_path   = article_dir / "article.md"
        img_path  = article_dir / "illustration.png"
        meta_path = article_dir / "meta.json"

        if not md_path.exists():
            continue

        # Charger article
        with md_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Charger métadonnées si disponibles
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            title  = meta.get("title", "Article sans titre")
            date   = meta.get("datetime", article_dir.name)
            sim    = meta.get("similarity", "?")
            labels = meta.get("labels", [])
        else:
            title  = article_dir.name
            date   = article_dir.name
            sim    = "?"
            labels = []

        # Affichage
        st.markdown("---")
        st.markdown(f"### 📝 {title}")
        st.caption(f"📅 Généré le : {date} · 🔎 Similarité : {sim} · ⚠️ Labels : {', '.join(labels) if labels else 'aucun'}")
        st.markdown(content, unsafe_allow_html=True)

        # Image si disponible
        if img_path.exists():
            st.image(str(img_path), width=512, caption="Illustration")