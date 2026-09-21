import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.models.baseline import plot_and_save_confusion_matrix

# Canonical label ordering matching Phase 2 exactly
CANONICAL_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

LABEL2ID = {label: i for i, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(CANONICAL_LABELS)}

DEFAULT_MAX_LENGTH = 256
BASELINE_VAL_MACRO_F1 = 0.7392
BASELINE_TEST_MACRO_F1 = 0.7165


def get_canonical_label_mapping():
    """Returns (label2id, id2label) deterministic mapping."""
    return dict(LABEL2ID), dict(ID2LABEL)


def calculate_class_weights(train_labels: pd.Series) -> torch.FloatTensor:
    """Calculates balanced class weights from training labels ONLY."""
    counts = train_labels.value_counts()
    total_samples = len(train_labels)
    num_classes = len(CANONICAL_LABELS)
    weights = []
    for label in CANONICAL_LABELS:
        cnt = counts.get(label, 0)
        w = total_samples / (num_classes * cnt) if cnt > 0 else 1.0
        weights.append(w)
    return torch.FloatTensor(weights)


class MentalHealthDataset(Dataset):
    """PyTorch Dataset wrapping tokenized encodings and labels."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            v = val[idx]
            if isinstance(v, torch.Tensor):
                item[key] = v.clone().detach()
            else:
                item[key] = torch.tensor(v)
        if self.labels is not None:
            lbl = self.labels[idx]
            if isinstance(lbl, torch.Tensor):
                item["labels"] = lbl.clone().detach().long()
            else:
                item["labels"] = torch.tensor(lbl, dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class WeightedLossTrainer(Trainer):
    """Custom Trainer overriding compute_loss to apply weighted CrossEntropyLoss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else outputs)

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Computes evaluation metrics for Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def build_detailed_metrics(labels, preds) -> dict:
    """Builds detailed metrics including per-class scores and confusion matrix."""
    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CANONICAL_LABELS))), average=None, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(CANONICAL_LABELS):
        per_class[label] = {
            "precision": float(per_p[idx]),
            "recall": float(per_r[idx]),
            "f1": float(per_f1[idx]),
            "support": int(per_sup[idx]),
        }

    cm = confusion_matrix(labels, preds, labels=list(range(len(CANONICAL_LABELS))))

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def predict_statement(
    statement: str,
    model,
    tokenizer,
    label2id=None,
    id2label=None,
    max_length=DEFAULT_MAX_LENGTH,
):
    """Predicts mental health status label, confidence score, and probability distribution."""
    if label2id is None or id2label is None:
        label2id, id2label = get_canonical_label_mapping()
    else:
        id2label = {int(k): v for k, v in id2label.items()}

    model.eval()
    inputs = tokenizer(
        statement,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    pred_prob = float(probs[pred_id])

    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "statement": statement,
        "predicted_label": pred_label,
        "prediction_probability": pred_prob,
        "probabilities": prob_dict,
    }


def get_resume_checkpoint(checkpoints_dir: str):
    """Returns the latest Hugging Face Trainer checkpoint path, if any."""
    if not os.path.isdir(checkpoints_dir):
        return None

    checkpoint_dirs = []
    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(path):
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            checkpoint_dirs.append((step, path))

    if not checkpoint_dirs:
        return None

    checkpoint_dirs.sort(key=lambda item: item[0])
    return checkpoint_dirs[-1][1]


def save_label_mapping(output_dir: str, label2id: dict, id2label: dict):
    """Persists canonical label mapping JSON."""
    mapping_path = os.path.join(output_dir, "label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)
    return mapping_path


def save_training_metadata(
    output_dir: str,
    training_config: dict,
    class_weights: torch.FloatTensor,
    validation_metrics: dict,
    test_metrics: dict | None = None,
):
    """Saves complete MentalBERT training metadata for reproducibility."""
    metadata = {
        **training_config,
        "canonical_label_ordering": CANONICAL_LABELS,
        "class_weights": {
            CANONICAL_LABELS[i]: float(class_weights[i]) for i in range(len(CANONICAL_LABELS))
        },
        "validation_best_metrics": validation_metrics,
        "baseline_comparison": {
            "linear_svc_validation_macro_f1": BASELINE_VAL_MACRO_F1,
            "linear_svc_test_macro_f1": BASELINE_TEST_MACRO_F1,
        },
    }
    if test_metrics is not None:
        metadata["test_metrics"] = test_metrics

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def tokenize_splits(tokenizer, train_df, val_df, test_df, max_length=DEFAULT_MAX_LENGTH):
    """Tokenizes train/validation/test splits with a shared max_length."""
    train_encodings = tokenizer(
        train_df["statement"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    val_encodings = tokenizer(
        val_df["statement"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_df["statement"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return train_encodings, val_encodings, test_encodings


def run_local_smoke_test(data_dir="data/processed/splits", output_dir="models/mental_bert"):
    """Executes a lightweight local CPU verification test without running full training."""
    print("=" * 70)
    print("RUNNING LOCAL SMOKE TEST (CPU Micro-Batch Verification - MentalBERT)")
    print("=" * 70)

    print("1. Verifying Environment & Packages:")
    print(f"   - Python version: {sys.version.split()[0]}")
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Data splits not found in {data_dir}. Ensure Phase 2 splits exist.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("\n2. Verified Data Splits:")
    print(f"   - Train shape: {train_df.shape} (Expected: 40844)")
    print(f"   - Val shape:   {val_df.shape} (Expected: 5105)")
    print(f"   - Test shape:  {test_df.shape} (Expected: 5106)")

    assert len(train_df) == 40844, "Train split size mismatch!"
    assert len(val_df) == 5105, "Validation split size mismatch!"
    assert len(test_df) == 5106, "Test split size mismatch!"

    label2id, id2label = get_canonical_label_mapping()
    print("\n3. Canonical Label Mapping:")
    print("   ", label2id)

    class_weights = calculate_class_weights(train_df["status"])
    print("\n4. Class Weights Calculated from Train Labels:")
    for idx, label in enumerate(CANONICAL_LABELS):
        print(f"   - {label:20s}: {class_weights[idx]:.4f}")

    print("\n5. Initializing Tokenizer & MentalBERT Model...")
    model_name = "mental/mental-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CANONICAL_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    print("\n6. Micro-Batch Forward Pass & Loss Test (5 samples)...")
    sample_texts = train_df["statement"].head(5).tolist()
    sample_labels = [label2id[l] for l in train_df["status"].head(5).tolist()]

    encodings = tokenizer(sample_texts, truncation=True, padding=True, max_length=DEFAULT_MAX_LENGTH)
    inputs = {k: torch.tensor(v) for k, v in encodings.items()}
    inputs["labels"] = torch.tensor(sample_labels, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, inputs["labels"])

    print(f"   - Forward Pass Logits Shape: {logits.shape}")
    print(f"   - Computed Micro-Batch Loss: {loss.item():.4f}")

    print("\n7. Sample Inference Test...")
    test_text = "I feel deeply anxious and overwhelmed by everything today."
    pred_res = predict_statement(test_text, model, tokenizer, label2id, id2label)
    print(f"   - Text: '{test_text}'")
    print(f"   - Predicted Class: {pred_res['predicted_label']}")
    print(f"   - Prediction Probability: {pred_res['prediction_probability']:.4f}")
    print("   - Top-3 Probabilities:")
    top_probs = sorted(pred_res["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
    for label, prob in top_probs:
        print(f"     * {label}: {prob:.4f}")

    smoke_save_path = os.path.join(output_dir, "smoke_test_model")
    os.makedirs(smoke_save_path, exist_ok=True)
    model.save_pretrained(smoke_save_path)
    tokenizer.save_pretrained(smoke_save_path)
    print(f"\n8. Checkpoint Save/Load Test at: {smoke_save_path}")

    loaded_model = AutoModelForSequenceClassification.from_pretrained(smoke_save_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(smoke_save_path)
    loaded_pred = predict_statement(test_text, loaded_model, loaded_tokenizer, label2id, id2label)

    assert loaded_pred["predicted_label"] == pred_res["predicted_label"], "Checkpoint load prediction mismatch!"
    assert np.isclose(
        loaded_pred["prediction_probability"],
        pred_res["prediction_probability"],
        rtol=1e-4,
        atol=1e-4,
    ), "Checkpoint load probability mismatch!"
    print("   - Reloaded model prediction matches saved checkpoint.")

    print("=" * 70)
    print("[PASS] LOCAL SMOKE TEST PASSED CLEANLY!")
    print("Full 3-epoch training was NOT performed locally.")
    print("Use Google Colab + CUDA GPU to execute full training.")
    print("=" * 70)


def run_full_training(
    data_dir="data/processed/splits",
    output_dir="models/mental_bert",
    num_epochs=3,
    batch_size=16,
    lr=2e-5,
    resume=False,
):
    """Executes full MentalBERT fine-tuning with CUDA GPU enforcement."""
    print("=" * 70)
    print("PHASE 3 — MENTALBERT FULL FINE-TUNING PIPELINE")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA GPU is unavailable.")
        print("Full MentalBERT fine-tuning requires a CUDA GPU (e.g., Google Colab T4).")
        print("Exiting full training safely without modifying model weights.\n")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA GPU Detected: {gpu_name}")
    print(f"PyTorch Version:  {torch.__version__}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    assert len(train_df) == 40844, "Train split size mismatch!"
    assert len(val_df) == 5105, "Validation split size mismatch!"
    assert len(test_df) == 5106, "Test split size mismatch!"

    label2id, id2label = get_canonical_label_mapping()
    class_weights = calculate_class_weights(train_df["status"])

    model_name = "mental/mental-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CANONICAL_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    print(f"\nTokenizing Train, Validation, and Test sets (max_length={DEFAULT_MAX_LENGTH})...")
    train_encodings, val_encodings, test_encodings = tokenize_splits(
        tokenizer, train_df, val_df, test_df, max_length=DEFAULT_MAX_LENGTH
    )

    train_labels = [label2id[l] for l in train_df["status"].tolist()]
    val_labels = [label2id[l] for l in val_df["status"].tolist()]
    test_labels = [label2id[l] for l in test_df["status"].tolist()]

    train_dataset = MentalHealthDataset(train_encodings, train_labels)
    val_dataset = MentalHealthDataset(val_encodings, val_labels)
    test_dataset = MentalHealthDataset(test_encodings, test_labels)

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    best_model_dir = os.path.join(output_dir, "best_model")
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=42,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        fp16=True,
        save_total_limit=3,
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    resume_from_checkpoint = None
    if resume:
        resume_from_checkpoint = get_resume_checkpoint(checkpoints_dir)
        if resume_from_checkpoint:
            print(f"\nResuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print("\nNo checkpoint found to resume; starting fresh training.")

    print("\nStarting MentalBERT fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("\nEvaluating best checkpoint on validation set...")
    val_pred = trainer.predict(val_dataset)
    val_metrics = build_detailed_metrics(val_pred.label_ids, np.argmax(val_pred.predictions, axis=1))
    print(f"   - Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"   - Validation Accuracy: {val_metrics['accuracy']:.4f}")

    print("\nSaving Best Model Artifacts...")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    save_label_mapping(output_dir, label2id, id2label)

    plot_and_save_confusion_matrix(
        val_metrics["confusion_matrix"],
        CANONICAL_LABELS,
        "Validation Confusion Matrix - MentalBERT",
        os.path.join(plots_dir, "cm_val_mental_bert.png"),
    )

    training_config = {
        "model_name": model_name,
        "max_length": DEFAULT_MAX_LENGTH,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_state": 42,
        "loss": "class_weighted_cross_entropy",
        "class_weight_source": "train_labels_only",
        "selection_metric": "validation_macro_f1",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "data_dir": data_dir,
        "output_dir": output_dir,
        "device": gpu_name,
    }

    metadata_path = save_training_metadata(
        output_dir=output_dir,
        training_config=training_config,
        class_weights=class_weights,
        validation_metrics=val_metrics,
    )

    print(f"Best model saved to: {best_model_dir}")
    print(f"Tokenizer saved to: {tokenizer_dir}")
    print(f"Metadata saved to: {metadata_path}")

    print("\nEvaluating selected best model ONCE on held-out test set...")
    test_pred = trainer.predict(test_dataset)
    test_metrics = build_detailed_metrics(test_pred.label_ids, np.argmax(test_pred.predictions, axis=1))
    print(f"   - Test Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"   - Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"   - Test Weighted F1: {test_metrics['weighted_f1']:.4f}")

    plot_and_save_confusion_matrix(
        test_metrics["confusion_matrix"],
        CANONICAL_LABELS,
        "Test Confusion Matrix - MentalBERT (Evaluated Once)",
        os.path.join(plots_dir, "cm_test_mental_bert.png"),
    )

    save_training_metadata(
        output_dir=output_dir,
        training_config=training_config,
        class_weights=class_weights,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    sample_text = "I feel deeply anxious and overwhelmed by everything today."
    sample_pred = predict_statement(sample_text, trainer.model, tokenizer, label2id, id2label)
    print("\nSample Inference (best model):")
    print(f"   - Text: '{sample_text}'")
    print(f"   - Predicted Class: {sample_pred['predicted_label']}")
    print(f"   - Prediction Probability: {sample_pred['prediction_probability']:.4f}")

    print("\nAll MentalBERT artifacts saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MentalBERT Mental Health Classifier")
    parser.add_argument("--smoke-test", action="store_true", help="Run lightweight local CPU smoke test")
    parser.add_argument("--full-train", action="store_true", help="Run full fine-tuning (requires CUDA GPU)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir/checkpoints")
    parser.add_argument("--data-dir", default="data/processed/splits", help="Directory containing CSV splits")
    parser.add_argument("--output-dir", default="models/mental_bert", help="Output directory for model artifacts")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    if args.smoke_test:
        run_local_smoke_test(data_dir=args.data_dir, output_dir=args.output_dir)
    elif args.full_train:
        run_full_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
        )
    else:
        run_local_smoke_test(data_dir=args.data_dir, output_dir=args.output_dir)
