"""
Carga y preparación del dataset AG News para topic classification.
"""
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_ag_news(config, seed=None):
    seed = seed or config["seed"]
    dataset_name = config["data"]["dataset_name"]

    ds = load_dataset(dataset_name)

    train = ds["train"].shuffle(seed=seed).select(
        range(config["data"]["train_subsample"])
    )
    test = ds["test"].shuffle(seed=seed).select(
        range(config["data"]["test_subsample"])
    )
    return train, test


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset, tokenizer, max_length):
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized
