import os

HF_TRAIN = "HF_TRAIN"
HF_EVALUATE = "HF_EVALUATE"
OPENAI_GENERATE = "OPENAI_GENERATE"
OPENAI_EVALUATE = "OPENAI_EVALUATE"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

