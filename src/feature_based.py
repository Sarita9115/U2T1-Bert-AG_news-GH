
"""
Feature-based adaptation: BERT/DistilBERT congelado como extractor de
features + regresión logística como clasificador.
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression


def load_frozen_encoder(model_name, device):
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def extract_features(model, tokenized_dataset, device, pooling="cls", batch_size=32):
    """Pasa el dataset tokenizado por el modelo congelado y devuelve
    embeddings (X) y labels (y) como arrays de numpy."""
    loader = DataLoader(tokenized_dataset, batch_size=batch_size)

    all_features = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if pooling == "cls":
            pooled = last_hidden[:, 0, :]  # token [CLS]
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            raise ValueError(f"Pooling desconocido: {pooling}")

        all_features.append(pooled.cpu().numpy())
        all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def run_feature_based(model_name, train_tok, test_tok, device, pooling="cls"):
    """Pipeline completo: extrae features y entrena logistic regression.
    Devuelve el clasificador entrenado + métricas de tiempo."""
    model = load_frozen_encoder(model_name, device)

    t0 = time.time()
    X_train, y_train = extract_features(model, train_tok, device, pooling)
    X_test, y_test = extract_features(model, test_tok, device, pooling)
    feature_time = time.time() - t0

    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    return {
        "classifier": clf,
        "X_test": X_test,
        "y_test": y_test,
        "feature_extraction_time_sec": feature_time,
        "classifier_train_time_sec": train_time,
        "n_trainable_params_body": 0,
        "n_trainable_params_head": X_train.shape[1] * len(set(y_train)),  # aprox
    }
