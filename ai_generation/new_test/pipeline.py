# pipeline.py
import os, json, datetime as dt
from pathlib import Path
import argparse

from article_generator import ArticleGenerator
from summarization import TextSummarizer
from similarity import SimilarityChecker
from ethical_filter_v2 import ethical_filter
from image_gen import ImageGenerator

# Chargement unique des modèles
gen   = ArticleGenerator()
summ  = TextSummarizer()
simch = SimilarityChecker()
imgg  = ImageGenerator(steps=15)

def run_pipeline(topic: str, out_dir: str = "outputs", with_image: bool = True) -> None:
    print(f"📌 Sujet : {topic}")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    article_dir = Path(out_dir) / timestamp
    article_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Génération de l'article…")
    article = gen.generate(topic)
    if article.lower().count("sexy") > 5 or article.count("Article:") > 5:
        raise ValueError("⚠️ Article incohérent détecté.")

    print("📚 Résumé…")
    summary = summ.summarize(article)

    print("📏 Similarité article/résumé…")
    sim_score = simch.compare(article, summary)

    print("🔍 Filtrage éthique…")
    filt = ethical_filter(summary)

    img_filename = None
    if with_image:
        print("🎨 Génération de l’image…")
        img_path = article_dir / "illustration.png"
        prompt = f"{topic}, vector illustration, flat design, clean lines, vibrant colors"
        imgg.generate(prompt, negative="text, watermark, lowres, distorted, blurry", path=str(img_path))
        img_filename = "illustration.png"

    print("💾 Sauvegarde Markdown…")
    with (article_dir / "article.md").open("w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(article + "\n\n---\n")
        f.write(f"## Résumé\n{summary}\n\n")
        f.write(f"### Similarité : **{sim_score:.3f}**\n\n")
        f.write("### Filtrage éthique\n")
        if filt["flagged"]:
            f.write(f"⚠️ Contenu sensible : {', '.join(filt['labels'])}\n\n")
            f.write("```json\n" + json.dumps(filt["scores"], indent=2, ensure_ascii=False) + "\n```\n")
        else:
            f.write("✅ Aucun contenu problématique détecté.\n\n")
        if img_filename:
            f.write(f"![Illustration]({img_filename})\n")

    print("💾 Sauvegarde meta.json…")
    meta = {
        "title": topic,
        "datetime": timestamp,
        "summary": summary,
        "similarity": round(sim_score, 3),
        "flagged": filt["flagged"],
        "labels": filt["labels"],
        "image": img_filename
    }
    with (article_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ Article généré : {article_dir.resolve()}")

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="*", help="Sujet de l'article")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--no-image", action="store_true", help="Désactiver la génération d'image")
    args = parser.parse_args()

    topic_text = " ".join(args.topic).strip() or "Les avancées récentes de l'IA"
    try:
        run_pipeline(topic_text, args.out_dir, with_image=not args.no_image)
    except Exception as e:
        print(f"❌ Erreur : {e}")
