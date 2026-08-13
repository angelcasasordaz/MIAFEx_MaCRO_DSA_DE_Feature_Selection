# -*- coding: utf-8 -*-
"""
Train MIAFEx (ViT backbone + element-wise refinement) for classification.

Outputs under output_dir:
  - miafex_checkpoint.pth
  - class_to_idx.json
  - metrics_curve.pkl
  - loss_and_combined_metrics_curve.png
"""
import argparse
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from miafex_model import MIAFEx


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


def _save_training_plot(loss_curve, acc_curve, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve, label="Training Loss")
    plt.plot(acc_curve, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("MIAFEx - Loss/Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_and_combined_metrics_curve.png"), dpi=160)
    plt.close()


def train_miafex(
    train_root,
    output_dir,
    num_classes=None,
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5,
    device=None,
):
    """
    Train MIAFEx and save the checkpoint, class mapping, metric curves, and plot.

    Returns the path to miafex_checkpoint.pth.
    """
    os.makedirs(output_dir, exist_ok=True)

    transform = _default_transform()
    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    if len(train_dataset) == 0:
        raise ValueError(f"No training images found in: {os.path.abspath(train_root)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    inferred_classes = len(train_dataset.classes)
    if num_classes is None:
        num_classes = inferred_classes
    elif inferred_classes != num_classes:
        print(
            f"[train] Warning: num_classes={num_classes} but dataset has "
            f"{inferred_classes}. Using {inferred_classes}."
        )
        num_classes = inferred_classes

    class_to_idx_path = os.path.join(output_dir, "class_to_idx.json")
    with open(class_to_idx_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, indent=2, ensure_ascii=False)

    resolved_device = _resolve_device(device)
    model = MIAFEx(num_classes=int(num_classes)).to(resolved_device)
    print(f"[train] device={resolved_device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_curve = []
    acc_curve = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        all_pred = []
        all_lbl = []

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
        for images, labels in pbar:
            images = images.to(resolved_device)
            labels = labels.to(resolved_device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_pred.append(logits.argmax(1).detach().cpu().numpy())
            all_lbl.append(labels.detach().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_lbl)
        acc = accuracy_score(y_true, y_pred)

        loss_curve.append(avg_loss)
        acc_curve.append(acc)
        print(f"[train] epoch={epoch + 1}  loss={avg_loss:.4f}  acc={acc:.4f}")

    with open(os.path.join(output_dir, "metrics_curve.pkl"), "wb") as f:
        pickle.dump({"loss": loss_curve, "accuracy": acc_curve}, f)

    _save_training_plot(loss_curve, acc_curve, output_dir)

    ckpt_path = os.path.join(output_dir, "miafex_checkpoint.pth")
    torch.save(
        {
            "vit_state_dict": model.vit.state_dict(),
            "fc_state_dict": model.fc.state_dict(),
            "refinement_weights": model.refinement_weights.detach().cpu(),
            "num_classes": int(num_classes),
        },
        ckpt_path,
    )
    print(f"[train] checkpoint saved: {ckpt_path}")
    return ckpt_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Train MIAFEx on an ImageFolder training split.")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_miafex(
        train_root=args.train_root,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
