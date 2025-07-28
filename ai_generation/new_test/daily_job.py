from pathlib import Path
from datetime import datetime
from article_generator import ArticleGenerator
from summarization      import TextSummarizer
from image_gen          import ImageGenerator

TOPICS = ["Impact IA sur la santé", "Python 3.13 en 5 points"]
outdir = Path("auto_posts")

def run():
    topic = TOPICS[datetime.now().day % len(TOPICS)]
    art   = ArticleGenerator().generate(topic)
    summ  = TextSummarizer().summarize(art)
    img   = ImageGenerator().generate(topic, path=outdir/"img.png")

    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir/"post.md", "w", encoding="utf8") as f:
        f.write(f"# {topic}\n\n{art}\n\n---\n**Résumé :** {summ}\n")
        f.write(f"\n![Image](img.png)\n")

if __name__ == "__main__":
    import schedule, time
    schedule.every().day.at("07:30").do(run)
    while True:
        schedule.run_pending()
        time.sleep(60)