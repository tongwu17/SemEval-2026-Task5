# SemEval 2026 Task 5

## Overview

[SemEval 2026 Task 5](https://nlu-lab.github.io/semeval.html): Predict the plausibility of a word sense in ambiguous sentences, scored on a 1–5 scale.

**Metrics:** Spearman Correlation, Accuracy Within Standard Deviation.

## Publications

- Paper: "NCL-UoR at SemEval-2026 Task 5: Embedding-Based Methods, Fine-Tuning, and LLMs for Word Sense Plausibility Rating" (https://aclanthology.org/2026.semeval-1.242.pdf)
- ACL Anthology: https://aclanthology.org/2026.semeval-1.242/
- DOI: [10.18653/v1/2026.semeval-1.242](https://doi.org/10.18653/v1/2026.semeval-1.242)
- Accepted at the Proceedings of the 20th International Workshop on Semantic Evaluation (2026)
- Publisher: Association for Computational Linguistics

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{wu-etal-2026-ncl-uor,
    title = "{NCL}-{U}o{R} at {S}em{E}val-2026 Task 5: Embedding-Based Methods, Fine-Tuning, and {LLM}s for Word Sense Plausibility Rating",
    author = "Wu, Tong  and
      Markchom, Thanet  and
      Liang, Huizhi(elly)",
    booktitle = "Proceedings of the 20th {I}nternational {W}orkshop on {S}emantic {E}valuation (2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.semeval-1.242/",
    doi = "10.18653/v1/2026.semeval-1.242",
    pages = "1930--1937",
    ISBN = "979-8-89176-414-9",
}
```

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
# Install dependencies (choose based on the method you want to run)
pip install -r src/requirements.txt              # src/methods/
pip install -r finetune_deberta/requirements.txt  # finetune_deberta/

# Set OpenAI API key (required for LLM method)
export OPENAI_API_KEY="your-key-here"

# Run a method (example)
python src/methods/llm_prompting.py
# Evaluate predictions 
cd semeval26-05-scripts
python scoring.py data/dev.json path/to/predictions.jsonl output/scores.json
```
