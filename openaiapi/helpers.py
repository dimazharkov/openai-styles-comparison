import json
import random


def get_dataset_from_file(file_path: str, count: int = 5):
    with open(file_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)

    return [item['description'] for item in data[:count]]
