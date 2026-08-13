# -*- coding: utf-8 -*-
"""
Extract refined MIAFEx descriptors from a trained checkpoint.

Outputs under output_dir:
  - extracted_features.csv
  - miafex_features.npy
  - class_to_idx.json

Optional ML baseline artifacts are written under output_dir/ml_eval when
run_ml_baselines=True.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from miafex_model import (
    MIAFEx,
    infer_num_classes_from_checkpoint,
    load_miafex_checkpoint,
    safe_load_checkpoint,
)


def _default_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_ml_baselines(X, y, output_dir):
    if len(np.unique(y)) < 2 or len(y) < 10:
        print("[extract] Skipping quick ML: need at least 2 classes and >= 10 samples.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    ml_dir = os.path.join(output_dir, "ml_eval")
    os.makedirs(ml_dir, exist_ok=True)

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {pr * 100:.2f}%  Recall: {rc * 100:.2f}%  F1: {f1 * 100:.2f}%")
        print(f"  Confusion matrix:\n{cm}")

        metrics = {
            "accuracy": float(acc),
            "precision_weighted": float(pr),
            "recall_weighted": float(rc),
            "f1_weighted": float(f1),
        }
        safe_name = name.replace(" ", "_").lower()
        with open(os.path.join(ml_dir, f"{safe_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(ml_dir, f"{safe_name}_report.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred))

        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(ml_dir, f"{safe_name}_confusion_matrix.png"), dpi=160)
        plt.close()

    print(f"[extract] ML artifacts saved to: {os.path.abspath(ml_dir)}")


def extract_miafex_features(
    data_dir,
    checkpoint_path,
    output_dir,
    batch_size=16,
    device=None,
    run_ml_baselines=False,
):
    """
    Extract 768-dimensional refined MIAFEx features and save CSV/NPY artifacts.

    Returns the path to extracted_features.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(checkpoint_path)}")

    resolved_device = _resolve_device(device)
    print(f"[extract] device={resolved_device}")

    dataset = datasets.ImageFolder(root=data_dir, transform=_default_transform())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with open(os.path.join(output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, indent=2, ensure_ascii=False)

    ckpt = safe_load_checkpoint(checkpoint_path, resolved_device)
    num_classes = infer_num_classes_from_checkpoint(ckpt, fallback=len(dataset.classes))
    if num_classes is None:
        raise ValueError("Could not infer num_classes from checkpoint or dataset.")

    model = MIAFEx(num_classes=int(num_classes)).to(resolved_device)
    loaded_num_classes = load_miafex_checkpoint(model, ckpt, resolved_device)
    print(f"[extract] num_classes={loaded_num_classes or num_classes}")

    model.eval()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting MIAFEx features"):
            images = images.to(resolved_device)
            _, refined = model(images)
            all_features.append(refined.detach().cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.vstack(all_features) if all_features else np.zeros((0, 768), dtype=np.float32)
    y = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)

    if X.shape[1] != 768:
        raise ValueError(f"Expected 768-dimensional refined features, got {X.shape[1]}.")

    csv_path = os.path.join(output_dir, "extracted_features.csv")
    features_npy = os.path.join(output_dir, "miafex_features.npy")

    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv(csv_path, index=False)
    np.save(features_npy, X)

    print(f"[extract] Features saved:")
    print(f"  CSV: {os.path.abspath(csv_path)}")
    print(f"  NPY: {os.path.abspath(features_npy)}")

    if run_ml_baselines:
        _run_ml_baselines(X, y, output_dir)

    return csv_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Extract MIAFEx features from an ImageFolder split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-ml-baselines", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_miafex_features(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        run_ml_baselines=args.run_ml_baselines,
    )
