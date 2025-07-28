"""
pipeline.py
Génère un article, son résumé, calcule la similarité, filtre le résumé,
crée une image d’illustration, puis sauvegarde le tout dans un répertoire.

Usage CLI :
    python pipeline.py "Sujet de l'article" --out_dir=outputs --no-image --temp 0.7
"""

import os, argparse, datetime as dt
from pathlib import Path

from article_generator import ArticleGenerator
from summarization     import TextSummarizer        # même classe que ton fichier
from similarity        import SimilarityChecker
from ethical_filter_v2 import ethical_filter       # ou ethical_filter (mono)
from image_gen         import ImageGenerator

# ──────────────────────────────────────────────
# 0. Instanciation unique
# ──────────────────────────────────────────────
article_gen = ArticleGenerator()         # distilgpt2
sum_gen     = TextSummarizer()           # distilbart-xsum
sim_checker = SimilarityChecker()
img_gen     = ImageGenerator()           # Stable Diffusion Turbo

# ──────────────────────────────────────────────
# 1. Fonction principale
# ──────────────────────────────────────────────
def run(topic: str,
        out_dir: str = "outputs",
        temperature: float = 0.8,
        with_image: bool = True) -> None:
    """Exécute le pipeline complet et écrit out_dir/article_<date>.md"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Article
    print("🧠 Génération de l'article…")
    article = article_gen.generate(topic, temperature=temperature)

    # 2. Résumé
    print("📚 Résumé…")
    summary = sum_gen.summarize(article)

    # 3. Similarité
    sim_score = sim_checker.compare(article, summary)

    # 4. Filtrage
    filt = ethical_filter(summary)

    # 5. Image optionnelle
    img_path_rel = ""
    if with_image:
        print("🎨 Image…")
        img_file = f"img_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
        img_path_abs = Path(out_dir) / img_file
        img_gen.generate(topic, path=str(img_path_abs))
        img_path_rel = f"![Illustration]({img_file})"

    # 6. Sauvegarde Markdown
    md_file = Path(out_dir) / f"article_{dt.datetime.now():%Y%m%d_%H%M%S}.md"
    with md_file.open("w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(article + "\n\n---\n")
        f.write(f"## Résumé\n{summary}\n\n")
        f.write(f"### Similarité (article / résumé) : **{sim_score:.3f}**\n\n")
        f.write("### Filtrage éthique\n")
        if filt["flagged"]:
            labels = ", ".join(filt["labels"])
            f.write(f"⚠️ Contenu sensible détecté : **{labels}**\n\n")
            f.write("```json\n" + str(filt["scores"]) + "\n```\n\n")
        else:
            f.write("✅ Aucun contenu problématique détecté.\n\n")
        if img_path_rel:
            f.write(img_path_rel + "\n")

    print(f"✅ Fichier écrit : {md_file}")

# ──────────────────────────────────────────────
# 2. Interface CLI très simple
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-pipeline blog.")
    parser.add_argument("topic", nargs="*", help="Sujet de l'article")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature GPT-2")
    parser.add_argument("--no-image", action="store_true", help="Désactive l'image")
    args = parser.parse_args()

    topic_text = " ".join(args.topic) or "Les bénéfices du sommeil sur la productivité"
    run(topic_text,
        out_dir=args.out_dir,
        temperature=args.temp,
        with_image=not args.no_image)