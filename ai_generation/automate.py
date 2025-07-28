from data_loader import load_sample_imdb_dataset
from ethical_filter import ethical_filter
from hackathon2.ai_generation.generation import TextGenerator
from summarization import TextSummarizer
from similarity import SimilarityChecker
from loguru import logger
import datetime

logger.add("runs/pipeline.log", rotation="500 KB", serialize=True)  # JSON log file

def run_pipeline(nb_exemples=3):
    dataset   = load_sample_imdb_dataset(sample_size=nb_exemples)
    generator = TextGenerator(max_length=80)
    summarizer = TextSummarizer()
    checker    = SimilarityChecker()

    logger.info(f"PIPELINE DÉMARRÉ — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    for i, example in enumerate(dataset):
        prompt = example["text"][:100].replace('\n', ' ').strip()
        generated = generator.generate(prompt)
        summary   = summarizer.summarize(generated)
        score     = checker.compare(prompt, summary)
        ethics    = ethical_filter(generated)

        logger.success(f"Extrait {i+1} | Sim={score:.3f} | Ethic={ethics['status']}")
        if ethics["flagged"]:
            logger.warning(f"Toxic label={ethics['label']} score={ethics['score']:.2f}")

    logger.info("Pipeline terminé.")

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