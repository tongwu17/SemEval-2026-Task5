# SemEval 2026 Task 5: Narrative Understanding

## Overview

[SemEval 2026 Task 5](https://nlu-lab.github.io/semeval.html): Predict the plausibility of a word sense in ambiguous sentences, scored on a 1–5 scale.

**Metrics:** Spearman Correlation, Accuracy Within Standard Deviation.

## Project Structure

```
├── finetune_deberta/              # DeBERTa fine-tuning experiments
├── plot/                          # Visualization
├── prompting/                     # Prompt-based approach
├── semeval26-05-scripts/          # Evaluation scripts
│   ├── data/                      # Train/dev/test data
│   ├── evaluate.py
│   ├── format_check.py
│   └── scoring.py
│
├── src/                           # Main implementation
│   ├── methods/
│   │   ├── embedding_mpnet_ridge.py
│   │   ├── embedding_roberta_xgboost.py
│   │   ├── llm_prompting.py
│   │   ├── finetuning_electra_base.py
│   │   └── finetuning_electra_lora.py
│   ├── config.py
│   ├── data_utils.py
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r src/requirements.txt

# Set OpenAI API key (required for LLM method)
export OPENAI_API_KEY="your-key-here"

# Run a method (example)
python src/methods/llm_prompting.py

# Evaluate predictions 
cd semeval26-05-scripts
python scoring.py data/dev.json path/to/predictions.jsonl output/scores.json
```
