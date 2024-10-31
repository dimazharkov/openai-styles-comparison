import json
import os
import random
import time
from typing import List

from constants import DATA_PATH
from openaiapi.client import use_azure_openai_client
from openaiapi.helpers import get_dataset_from_file


def discriminate():
    epochs = 10
    dataset_rate_sum = failed_rate_sum = 0

    for i in range(epochs):
        dataset_rate, failed_rate = evaluate_epoch()
        dataset_rate_sum += dataset_rate
        failed_rate_sum += failed_rate

    print(f"Epochs: {epochs}, gen rate: {(dataset_rate_sum / epochs):.1f}, failed rate: {(failed_rate_sum / epochs):.1f}")


def evaluate_epoch():
    generated_set = get_dataset_from_file(
        file_path = os.path.join(DATA_PATH, "generated.json")
    )

    dataset_rate = evaluate_dataset(
        generated_set
    )

    print(f"Ratio of generated: {dataset_rate}/{len(generated_set)+1}")

    failed_set = get_failed_set(
        file_path = os.path.join(DATA_PATH, "test.json")
    )

    failed_rate = evaluate_dataset(
        generated_set
    )

    print(f"Failed rate: {failed_rate}/{len(failed_set)+1}")

    return dataset_rate, failed_rate


def evaluate_dataset(dataset: List[str]) -> int:
    corpus_subset = get_dataset_from_file(
        file_path = os.path.join(DATA_PATH, "corpus.json")
    )

    for item in dataset:
        corpus_subset = evaluate_item(
            corpus_subset, item
        )
        time.sleep(1)

    intersection = set(corpus_subset) & set(dataset)
    return len(intersection)


def evaluate_item(corpus: List[str], generated_item: str) -> List[str]:
    prompt = (
        "You are an expert in analyzing text styles. I will give you six bouquet descriptions. "
        "Analyze them based on stylistic differences, such as tone, choice of words, and descriptive style. "
        "Identify the description that is most distinct from the others in terms of text style only. "
        "Return only the number of the most distinct bouquet as a single-digit number from 1 to 6, without any additional text."
    )

    corpus.append(generated_item)
    random.shuffle(corpus)

    text = f"The bouquet descriptions are:\n" + "\n".join(
        [f"{i + 1}. {desc}" for i, desc in enumerate(corpus)]
    )

    index = use_azure_openai_client(
        prompt=prompt, text=text, temperature=0.2
    )

    del corpus[int(index) - 1]

    return corpus


def get_failed_set(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    invalid_items = [item['description'] for item in data if not item.get("valid", True)]
    random.shuffle(invalid_items)

    return invalid_items[:5]