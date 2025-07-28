from data_loader import load_sample_imdb_dataset
from generation import TextGenerator
from summarization import TextSummarizer
from similarity import SimilarityChecker
from ethical_filter import ethical_filter  # Ajout du filtre

if __name__ == "__main__":
    dataset = load_sample_imdb_dataset(sample_size=5)
    generator = TextGenerator(max_length=80)
    summarizer = TextSummarizer()
    checker = SimilarityChecker()

    for i, example in enumerate(dataset):
        prompt = example["text"][:100].replace('\n', ' ').strip()
        generated = generator.generate(prompt)
        summary = summarizer.summarize(generated)
        similarity_score = checker.compare(prompt, summary)
        ethics = ethical_filter(generated)  # Application du filtre éthique

        print(f"\nExtrait {i+1}")
        print("Prompt (dataset IMDB)        :", prompt)
        print("Texte généré (distilGPT2)    :", generated)
        print("Résumé (distilBART-cnn-12-6) :", summary)
        print(f"Similarité prompt/résumé     : {similarity_score * 100:.2f}%")
        print(f"Filtrage éthique             : {ethics['status']}")
        if ethics["flagged"]:
            print("Mots détectés                :", ethics["keywords"])