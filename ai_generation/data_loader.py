from datasets import load_dataset
import random


def load_sample_imdb_dataset(sample_size=50, seed=42):
    """
    Charge et sous-échantillonne le dataset IMDB.

    Args:
        sample_size (int): nombre d'exemples à charger
        seed (int): graine aléatoire pour reproductibilité

    Returns:
        list of dict: exemples sous forme [{text: ..., label: ...}, ...]
    """

    dataset = load_dataset("imdb", split="train[:5%]")  # ~1250 exemples
    dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return [{"text": item["text"], "label": item["label"]} for item in dataset]