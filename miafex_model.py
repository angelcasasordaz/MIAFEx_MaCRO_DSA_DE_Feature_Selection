# -*- coding: utf-8 -*-
"""Shared MIAFEx model and checkpoint helpers."""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class MIAFEx(nn.Module):
    """ViT backbone + element-wise refinement for 768-dimensional descriptors."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            output_hidden_states=True,
        )
        self.fc = nn.Linear(768, num_classes)
        self.refinement_weights = nn.Parameter(torch.randn(768))

    def forward(self, x):
        vit_outputs = self.vit(x)
        if vit_outputs.hidden_states is None:
            raise ValueError("Hidden states not returned. Ensure output_hidden_states=True.")
        cls_features = vit_outputs.hidden_states[-1][:, 0, :]
        refined = cls_features * self.refinement_weights.view(1, -1)
        logits = self.fc(refined)
        return logits, refined


def safe_load_checkpoint(path: str, device):
    """Load a PyTorch checkpoint using weights_only when the local PyTorch supports it."""
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location=device)


def infer_num_classes_from_checkpoint(ckpt: dict, fallback: int | None = None) -> int | None:
    if "num_classes" in ckpt:
        return int(ckpt["num_classes"])
    if "fc_state_dict" in ckpt and "weight" in ckpt["fc_state_dict"]:
        return int(ckpt["fc_state_dict"]["weight"].shape[0])
    if "model_state_dict" in ckpt:
        weight = ckpt["model_state_dict"].get("fc.weight")
        if weight is not None:
            return int(weight.shape[0])
    return fallback


def load_miafex_checkpoint(model: MIAFEx, ckpt: dict, device) -> int | None:
    """Load ViT, FC, and refinement weights from train_miafex-compatible checkpoints."""
    if "vit_state_dict" in ckpt:
        vit_state_dict = ckpt["vit_state_dict"]
    elif "model_state_dict" in ckpt:
        vit_state_dict = {
            key.removeprefix("vit."): value
            for key, value in ckpt["model_state_dict"].items()
            if key.startswith("vit.")
        }
        if not vit_state_dict:
            raise KeyError("No 'vit.' keys found in 'model_state_dict'.")
    else:
        raise KeyError("Checkpoint must contain 'vit_state_dict' or 'model_state_dict' with 'vit.' keys.")

    missing, unexpected = model.vit.load_state_dict(vit_state_dict, strict=False)
    if missing:
        print(f"[miafex] Warning: missing keys in ViT: {missing}")
    if unexpected:
        print(f"[miafex] Warning: unexpected ViT keys ignored: {unexpected}")

    if "fc_state_dict" in ckpt:
        model.fc.load_state_dict(ckpt["fc_state_dict"], strict=True)
    elif "model_state_dict" in ckpt:
        fc_state_dict = {
            key.removeprefix("fc."): value
            for key, value in ckpt["model_state_dict"].items()
            if key.startswith("fc.")
        }
        if fc_state_dict:
            model.fc.load_state_dict(fc_state_dict, strict=True)

    if "refinement_weights" in ckpt:
        with torch.no_grad():
            refinement_weights = ckpt["refinement_weights"].to(device)
            if refinement_weights.numel() != model.refinement_weights.numel():
                raise ValueError(
                    "refinement_weights size mismatch: "
                    f"ckpt={refinement_weights.numel()} vs model={model.refinement_weights.numel()}"
                )
            model.refinement_weights.copy_(refinement_weights)

    return infer_num_classes_from_checkpoint(ckpt)
