# Artificial Intelligence Course Labs

Two reproducible artificial intelligence exercises:

1. Iris dataset classification with a neural network
2. Extractive question answering with Transformers

## What was improved

- Reusable functions replace monolithic scripts.
- Random seeds make runs easier to reproduce.
- Data normalization is fitted only on the training set.
- The Keras model uses an explicit `Input` layer.
- The Transformer model is loaded only when needed.
- The question-answering pipeline can be injected, so tests do not need to download a model.
- Both exercises have command-line arguments.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Iris classifier

```powershell
$env:PYTHONPATH='src'
python -m ai_labs.iris_classifier --epochs 60
```

## Question answering

```powershell
$env:PYTHONPATH='src'
python -m ai_labs.transformer_qa
```

The first run downloads the Hugging Face model. The default context and questions are in English because the model was fine-tuned on SQuAD2. You can choose another model with `--model`.

## Tests

The tests do not require network access:

```powershell
.\run-tests.ps1
```
