from data_loader import load_sample_imdb_dataset
from ethical_filter import ethical_filter
from generation import TextGenerator
from summarization import TextSummarizer
from similarity import SimilarityChecker
import datetime


def run_pipeline(nb_exemples=3):
    # Initialisation modules
    dataset = load_sample_imdb_dataset(sample_size=nb_exemples)
    generator = TextGenerator(max_length=80)
    summarizer = TextSummarizer()
    checker = SimilarityChecker()

    print(f"\n=== PIPELINE DÉMARRÉ — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    for i, example in enumerate(dataset):
        prompt = example["text"][:100].replace('\n', ' ').strip()
        generated = generator.generate(prompt)
        summary = summarizer.summarize(generated)
        score = checker.compare(prompt, summary)
        ethics = ethical_filter(generated)

        print(f"\n🔹 Extrait {i+1}")
        print("Prompt :", prompt)
        print("Généré :", generated)
        print("Résumé :", summary)
        print(f"Filtrage éthique : {ethics['status']}")
        if ethics["flagged"]:
            print(f"  Label : {ethics['label']}, Score : {ethics['score']:.2f}")

    print("\nPipeline terminé.")


if __name__ == "__main__":
    run_pipeline(nb_exemples=3)

"""     import schedule
import time
from automate import run_pipeline

# Planifier : toutes les 1 min
schedule.every(1).minutes.do(run_pipeline, nb_exemples=2)

print("Lancement planificateur. CTRL+C pour stopper.")
while True:
    schedule.run_pending()
    time.sleep(1) """