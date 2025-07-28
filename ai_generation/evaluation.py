import nltk
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
evaluate_adversarial()