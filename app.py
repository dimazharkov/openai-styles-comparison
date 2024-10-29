import os
from typing import Union
from analyzer import analyze
from trainer import train

def check_directory(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory '{path}' does not exist.")
    if not os.listdir(path):
        raise ValueError(f"Directory '{path}' is empty.")

def app(args: Union[list, None]):
    if args is not None and len(args) > 0:
        first_arg = args[0]
        if first_arg == 'train':
            train()
        else:
            print("Error: unknown argument")
        return

    try:
        check_directory('./model')
        check_directory('./output')
    except (FileNotFoundError, ValueError):
        print("Error: Please train the model first using: main.py train")
        return

    analyze()
