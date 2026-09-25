# U2T1 — Adapting BERT for NLP Tasks: Topic Classification (AG News)

Assignment U2T01: adapt BERT to four classical NLP tasks, comparing
adaptation methods (feature-based vs. fine-tuning) with empirical
measurements. This repo covers **Topic 1: Topic Classification** on the
AG News dataset (4 classes: World, Sports, Business, Sci/Tech).

## Project structure

```
├── BERT_AG_news_colab   # main notebook (run on Google Colab)
├── src/
│   ├── data.py            # dataset loading, subsampling, tokenization
│   ├── feature_based.py   # frozen encoder + logistic regression
│   ├── finetune.py        # partial fine-tuning (top layers + head, dual LR)
│   └── metrics.py         # accuracy, F1 macro, confusion matrix
├── configs/
│   └── config.yaml        # seeds, paths, hyperparameters
├── requirements.txt
└── README.md
```

`outputs/` (checkpoints, metrics JSON, figures) is intentionally not
versioned here — it's generated at runtime and persisted to Google Drive
instead. See "How to reproduce" below.

## Approach

Two adaptation methods were compared, per the assignment's ladder:

1. **Feature-based**: BERT body frozen (0 trainable params), `[CLS]`
   embedding from the last hidden state fed into a logistic regression
   classifier (scikit-learn).
2. **Partial fine-tuning**: top 2 encoder layers + classification head
   trainable (~14.7M params), two learning-rate groups (head: 1e-3,
   encoder: 2e-5), 2 epochs.

Both methods were first developed and debugged with **DistilBERT**
(faster iteration), then re-run with **bert-base-uncased** (the body
required by the assignment) to produce the final reported numbers and
delivered model.

## Results

| Method | Model | Trainable params | Accuracy | F1 macro |
|---|---|---|---|---|
| Feature-based | DistilBERT | 0 | 0.8850 | 0.8861 |
| Partial fine-tuning | DistilBERT | 14,769,412 | 0.9060 | 0.9070 |
| Feature-based | BERT-base | 0 | 0.8735 | 0.8742 |
| **Partial fine-tuning** | **BERT-base** | **14,769,412** | **0.9025** | **0.9035** |

Delivered model: partial fine-tuning on bert-base-uncased, published at
[Sarita9115/BERT-AG_news-HF](https://huggingface.co/Sarita9115/BERT-AG_news-HF).

## How to reproduce

1. Open `BERT_AG_news_colab` in Google Colab.
2. Mount Google Drive (for persisting outputs across sessions).
3. Clone this repo inside the Colab runtime:
   ```bash
   git clone https://github.com/Sarita9115/U2T1-Bert-AG_news-GH.git
   cd U2T1-Bert_in_AG_news
   pip install -r requirements.txt
   ```
4. Run the notebook cells in order. `configs/config.yaml` controls the
   dataset subsample size, model names (`dev`/`final`), and
   hyperparameters.

## References

- Devlin et al., 2018. [BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Tunstall, von Werra & Wolf. *Natural Language Processing with
  Transformers* (O'Reilly), chapters 1–3.
- Dataset: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
