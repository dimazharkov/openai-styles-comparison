import json
import os
import random

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from constants import DATA_PATH


def analyze():
    return test()


def test():
    tp = tn = fp = fn = 0
    model = GPT2LMHeadModel.from_pretrained('../model')
    tokenizer = GPT2Tokenizer.from_pretrained('../model')

    with open(os.path.join(DATA_PATH, "test.json"), 'r') as f:
        data = json.load(f)
        random.shuffle(data)

        for entry in data:
            perplexity = calc_perplexity(
                model, tokenizer, entry['description']
            )
            valid = True if perplexity <= 50 else False
            print('---')
            print(entry['description'])
            print(f"product: {entry['comment']}, label: {entry['valid']}, perplexity: {perplexity}, valid: {valid}")

            tp += valid and entry['valid']
            fn += valid and not entry['valid']
            fp += not valid and entry['valid']
            tn += not valid and not entry['valid']

    precision, recall, f1 = calc_metrics(tp, fp, fn, tn)

    print(f"--------- ")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")


def calc_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calc_perplexity(model, tokenizer, text) -> float:
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()

