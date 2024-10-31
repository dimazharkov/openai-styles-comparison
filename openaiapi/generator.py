import json
import os

from constants import DATA_PATH
from openaiapi.client import use_azure_openai_client
from openaiapi.helpers import get_dataset_from_file


def generate():
    descriptions = get_dataset_from_file(
        file_path=os.path.join(DATA_PATH, "corpus.json")
    )
    flowers_str = get_flowers_str()
    prompt = "You are an assistant that generates bouquet descriptions based on given examples."
    text = f"""
    I have the following bouquet descriptions:
    
    1. {descriptions[0]}
    2. {descriptions[1]}
    3. {descriptions[2]}
    4. {descriptions[3]}
    5. {descriptions[4]}
    
    The flowers used in these bouquets include: {flowers_str}. """ + """
    
    Please generate 5 new bouquet descriptions that are similar in style, tone, and structure to the examples provided. Each description should contain at least one of the listed flowers or a combination of them. Return the response as a valid JSON list with the following structure:
    
    [
        {"description": "New bouquet description 1"},
        {"description": "New bouquet description 2"},
        {"description": "New bouquet description 3"},
        {"description": "New bouquet description 4"},
        {"description": "New bouquet description 5"}
    ]    
    """

    response = use_azure_openai_client(
        prompt=prompt, text=text, temperature=0.7, max_tokens=500
    )

    save_to_file(
        response, file_path=os.path.join(DATA_PATH, "generated.json")
    )

    print(response)



def get_flowers_str():
    flowers = [
        "Tulip",
        "Rose",
        "Peony",
        "Lily",
        "Chrysanthemum",
        "Orchid",
        "Cornflower",
        "Hyacinth",
        "Lavender",
        "Carnation"
    ]

    return ", ".join(flowers)


def save_to_file(response, file_path):
    if isinstance(response, str):
        response = json.loads(response)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, ensure_ascii=False)
