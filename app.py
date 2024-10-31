import os

from constants import HF_TRAIN, HF_EVALUATE, OPENAI_GENERATE, OPENAI_EVALUATE
from huggingface.analyzer import analyze as huggingface_analyze
from huggingface.trainer import train as huggingface_train
from openaiapi.generator import generate as openai_generate
from openaiapi.discriminator import discriminate as openai_discriminate

def check_directory(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory '{path}' does not exist.")
    if not os.listdir(path):
        raise ValueError(f"Directory '{path}' is empty.")

def app(action):
    if action == HF_TRAIN:
        huggingface_train()
    elif action == HF_EVALUATE:
        try:
            check_directory('./model')
            check_directory('./output')
        except (FileNotFoundError, ValueError):
            print("Error: Please train the model first using: main.py train")
            return
        huggingface_analyze()
    elif action == OPENAI_GENERATE:
        openai_generate()
    elif action == OPENAI_EVALUATE:
        openai_discriminate()
    else:
        print("Unknown action.")
