import json
import os

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer


def train():
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)

    for param in model.transformer.parameters():
        param.requires_grad = False

    for param in model.lm_head.parameters():
        param.requires_grad = True

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = prep_dataset(
        tokenizer, './data/corpus.json'
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir='./output',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        save_steps=5000,
        learning_rate=1e-5,
        # weight_decay=0.01,
        evaluation_strategy="epoch",
    )

    split_dataset = dataset['train'].train_test_split(test_size=0.1)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
    )

    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')


def prep_dataset(tokenizer, file_path):
    dataset = load_dataset(
        "json", data_files=file_path
    )

    def encode(item):
        return tokenizer(
            item['description'], truncation=True, padding="max_length", max_length=256
        )

    tokenized_dataset = dataset.map(
        encode, batched=True
    )

    return tokenized_dataset



def prep_txt_dataset(tokenizer, file_path):
    return TextDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        block_size=128,
    )


def prep_corpus_file(file_path: str = './data/corpus.txt') -> None:
    if os.path.exists(file_path):
        return

    file_path_json = file_path.replace(".txt", ".json")
    if not os.path.exists(file_path_json):
        raise FileNotFoundError

    with (open(file_path_json, 'r', encoding='utf-8')) as f:
        data = json.load(f)

    with (open(file_path, 'w', encoding='utf-8')) as f:
        for entry in data:
            content = ' '.join(entry['description'])
            f.write(content + '\n')
