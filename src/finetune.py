
"""
Partial fine-tuning: BERT/DistilBERT con cabeza de clasificación,
congelando la mayoría del encoder y dejando entrenables solo las
últimas capas + la cabeza, con dos grupos de learning rate.
"""
import time
import numpy as np
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score


def build_model(model_name, num_labels, unfrozen_layers):
    """Carga el modelo y congela todo el encoder excepto las últimas
    `unfrozen_layers` capas (+ la cabeza, que siempre está entrenable
    porque se inicializa desde cero)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Detectar la lista de capas del encoder según arquitectura
    # (BERT: model.bert.encoder.layer, DistilBERT: model.distilbert.transformer.layer)
    if hasattr(model, "bert"):
        layers = model.bert.encoder.layer
        embeddings = model.bert.embeddings
    elif hasattr(model, "distilbert"):
        layers = model.distilbert.transformer.layer
        embeddings = model.distilbert.embeddings
    else:
        raise ValueError("Arquitectura no soportada por build_model")

    # Congelar embeddings siempre
    for param in embeddings.parameters():
        param.requires_grad = False

    # Congelar todas las capas del encoder excepto las últimas N
    n_layers = len(layers)
    for i, layer in enumerate(layers):
        requires_grad = i >= (n_layers - unfrozen_layers)
        for param in layer.parameters():
            param.requires_grad = requires_grad

    return model


def build_optimizer(model, lr_head, lr_encoder):
    """Arma el optimizer con dos grupos de parámetros: cabeza (LR alto)
    y capas superiores descongeladas del encoder (LR bajo)."""
    head_params = []
    encoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "pre_classifier" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = AdamW([
        {"params": head_params, "lr": lr_head},
        {"params": encoder_params, "lr": lr_encoder},
    ])
    return optimizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def run_finetune(model_name, train_tok, test_tok, config, output_dir):
    ft_cfg = config["finetune"]
    num_labels = 4

    model = build_model(model_name, num_labels, ft_cfg["unfrozen_layers"])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {n_trainable:,} / {n_total:,}")

    optimizer = build_optimizer(model, ft_cfg["lr_head"], ft_cfg["lr_encoder"])

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/finetune_checkpoints",
        num_train_epochs=ft_cfg["num_train_epochs"],
        per_device_train_batch_size=ft_cfg["batch_size"],
        per_device_eval_batch_size=ft_cfg["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),  # (optimizer, lr_scheduler=None)
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    eval_result = trainer.evaluate()

    return {
        "trainer": trainer,
        "model": model,
        "eval_result": eval_result,
        "train_time_sec": train_time,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
    }
