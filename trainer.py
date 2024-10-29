import json
import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer


def train():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    corpus_file_path = './data/corpus.txt'

    prep_corpus_file(corpus_file_path)

    dataset = TextDataset(
        file_path=corpus_file_path,
        tokenizer=tokenizer,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir='./output',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10000,
        save_total_limit=2,
        # learning_rate=2e-5,
        # weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')


def prep_corpus_file(file_path: str = './data/corpus.txt') -> None:
    if os.path.exists(file_path):
        return

    file_path_json = file_path.replace(".txt",".json")
    if not os.path.exists(file_path_json):
        raise FileNotFoundError

    with (open(file_path_json, 'r', encoding='utf-8')) as f:
        data = json.load(f)

    with (open(file_path, 'w', encoding='utf-8')) as f:
        for entry in data:
            content = ' '.join(entry['description'])
            f.write(content + '\n')
