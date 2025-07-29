""" import nltk
nltk.download('punkt')  # à exécuter une seule fois
from generation import TextGenerator
from summarization import TextSummarizer
from similarity import SimilarityChecker
from data_loader import load_sample_imdb_dataset

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

import numpy as np

def evaluate_pipeline(nb_exemples=10):
    dataset = load_sample_imdb_dataset(sample_size=nb_exemples)
    generator = TextGenerator(max_length=80)
    summarizer = TextSummarizer()
    checker = SimilarityChecker()

    bleu_scores = []
    rouge_scores = []
    similarity_scores = []

    for example in dataset:
        prompt = example["text"][:100].replace('\n', ' ').strip()
        generated = generator.generate(prompt)
        summary = summarizer.summarize(generated)
        similarity = checker.compare(prompt, summary)

        # BLEU
        bleu = sentence_bleu([prompt.split()], generated.split())
        bleu_scores.append(bleu)

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge = scorer.score(prompt, summary)["rougeL"].fmeasure
        rouge_scores.append(rouge)

        # Similarity
        similarity_scores.append(similarity)

    print("\n=== ÉVALUATION SUR", nb_exemples, "EXEMPLES ===")
    print(f"BLEU moyen      : {np.mean(bleu_scores):.4f}")
    print(f"ROUGE-L moyen   : {np.mean(rouge_scores):.4f}")
    print(f"Similarité moyenne : {np.mean(similarity_scores):.4f}")

def evaluate_adversarial(path="adversarial_prompts.txt"):
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    generator = TextGenerator(max_length=80)
    summarizer = TextSummarizer()
    checker = SimilarityChecker()

    print(f"\n=== TEST PROMPTS ADVERSARIAUX ({len(prompts)} prompts) ===")
    for i, prompt in enumerate(prompts):
        generated = generator.generate(prompt)
        summary = summarizer.summarize(generated)
        score = checker.compare(prompt, summary)

        print(f"\nPrompt {i+1} : {prompt}")
        print("Généré       :", generated)
        print("Résumé       :", summary)
        print(f"Similarité   : {score * 100:.2f}%")


evaluate_pipeline()
evaluate_adversarial() """

from article_generator import ArticleGenerator   # nouveau générateur léger
from summarization     import TextSummarizer     # même fichier, même classe
from similarity import SimilarityChecker
from data_loader import load_sample_imdb_dataset
from loguru import logger
from evaluate import load as load_metric
import numpy as np, json, sqlite3, os, torch

ppl_metric = load_metric("perplexity", module_type="metric", model_id="gpt2-medium")

def evaluate_pipeline(n=50, save_db="runs/history.sqlite"):
    ds = load_sample_imdb_dataset(sample_size=n)
    gen  = ArticleGenerator(max_new_tokens=400)
    sumr = TextSummarizer()
    sim = SimilarityChecker()

    results = []
    rouge_metric = load_metric("rouge")
    for ex in ds:
        prompt = ex["text"][:200].replace("\n", " ")
        generated = gen.generate(prompt)
        summary = sumr.summarize(generated)
        rouge = rouge_metric.compute(predictions=[summary], references=[generated])["rougeL"].mid.fmeasure
        ppl = ppl_metric.compute(predictions=[generated])["perplexities"][0]
        s = sim.compare(prompt, summary)
        results.append((rouge, ppl, s))

    r_mean = np.mean([r for r,_,_ in results])
    p_mean = np.mean([p for _,p,_ in results])
    s_mean = np.mean([s for *_,s in results])
    logger.success(f"ROUGE-L : {r_mean:.4f} | Perplexité : {p_mean:.2f} | Similarité : {s_mean:.4f}")

    # ---- Persistance ----
    os.makedirs("runs", exist_ok=True)
    conn = sqlite3.connect(save_db)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS evals (
        ts TEXT,
        n INT,
        rouge REAL,
        ppl REAL,
        sim REAL,
        model TEXT DEFAULT 'default'
    )""")
    cur.execute("INSERT INTO evals VALUES (datetime('now'),?,?,?,?,?)",
            (n, r_mean, p_mean, s_mean, "v1"))
    conn.commit(); conn.close()