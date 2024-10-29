
# Project Overview

This project focuses on **stylistic analysis of text** through the fine-tuning of a **GPT-2 model** on a custom text corpus. The goal is to assess whether a given text aligns with a specific style by evaluating the **perplexity** of the model on the input data.

## Key Features
- **Fine-tuning GPT-2**: The GPT-2 model is trained on a selected corpus to adapt it to specific stylistic features.
- **Style Matching via Perplexity**: The primary metric for evaluating the stylistic conformity of a text is **perplexity**. Lower perplexity indicates a higher match to the desired style.
- **Custom Text Corpus**: The model is fine-tuned using a corpus carefully curated to reflect the target style or genre.


## Usage

To use the application, you must first train the model if it has not been trained yet.

### Train the Model:
```bash
python main.py train
```

### Analyze Data:
Once the model is trained, run the analysis:

```bash
python main.py
```

The analyzer will process the texts from the ./data/test.json file.

## Usage
This project can be used for:
- **Text similarity analysis**: Checking how well a given text conforms to the style of a specific author or genre.
- **Plagiarism detection**: Identifying stylistic deviations in texts.
- **Content generation**: Enhancing text generation by fine-tuning on domain-specific styles.

## Requirements
- **Python 3.12+**
- **Hugging Face Transformers library**
- **PyTorch**
- A **corpus** of texts representing the desired style

## How It Works
1. Fine-tune the GPT-2 model on the corpus representing the desired style.
2. Evaluate the perplexity of the input text using the fine-tuned model.
3. Interpret perplexity: The lower the perplexity, the more the text matches the target style.

This approach leverages GPT-2's capacity to generate and analyze text, using **perplexity as a statistical measure** to gauge stylistic similarity.

## License

This project is licensed under the MIT License.
