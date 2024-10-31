import sys
from app import app
from constants import HF_TRAIN, HF_EVALUATE, OPENAI_GENERATE, OPENAI_EVALUATE

options = {
    '1': HF_TRAIN,
    '2': HF_EVALUATE,
    '3': OPENAI_GENERATE,
    '4': OPENAI_EVALUATE
}


def main():
    print("Please select one of the options:")
    print("1 - (Hugging Face) Train model")
    print("2 - (Hugging Face) Evaluate")
    print("3 - (OpenAI) Generate")
    print("4 - (OpenAI) Evaluate")

    choice = input("Enter your choice: ")

    if choice in ['1','2','3','4']:
        app(options[choice])
    else:
        print("Incorrect input. Please try again.")
        main()


if __name__ == '__main__':
    main()
