"""MEALPY optimizer acronym registry for the installed MEALPY version."""
from __future__ import annotations

from typing import Dict

import mealpy


CUSTOM_OPTIMIZERS = ["MaCRO-DE", "DSADE", "DSADE_AWAD", "DBO"]
CUSTOM_OPTIMIZER_KEYS = {name.upper().replace("_", "-"): name for name in CUSTOM_OPTIMIZERS}
CUSTOM_OPTIMIZER_KEYS["MACRO_DE"] = "MaCRO-DE"
CUSTOM_OPTIMIZER_KEYS["DSA-DE"] = "DSADE"
CUSTOM_OPTIMIZER_KEYS["DSADE-AWAD"] = "DSADE_AWAD"


def _installed_optimizer_names() -> set[str]:
    return set(mealpy.get_all_optimizers(verbose=False).keys())


def _build_mealpy_registry() -> Dict[str, str]:
    installed = _installed_optimizer_names()
    registry = {}
    for class_name in installed:
        if class_name.startswith("Original") and len(class_name) > len("Original"):
            acronym = class_name[len("Original"):].upper()
            registry.setdefault(acronym, class_name)

    preferred = {
        "PSO": "OriginalPSO",
        "GWO": "OriginalGWO",
        "WOA": "OriginalWOA",
        "DE": "OriginalDE",
        "HHO": "OriginalHHO",
        "FOX": "OriginalFOX",
        "RIME": "OriginalRIME",
        "RUN": "OriginalRUN",
    }
    for acronym, class_name in preferred.items():
        if class_name in installed:
            registry[acronym] = class_name
    return dict(sorted(registry.items()))


MEALPY_OPTIMIZER_REGISTRY = _build_mealpy_registry()
_MEALPY_CLASS_NAMES = _installed_optimizer_names()


def resolve_optimizer_name(name: str) -> str:
    raw = str(name)
    key = raw.upper()
    custom_key = key.replace("_", "-")
    if custom_key in CUSTOM_OPTIMIZER_KEYS:
        return CUSTOM_OPTIMIZER_KEYS[custom_key]
    if key in CUSTOM_OPTIMIZER_KEYS:
        return CUSTOM_OPTIMIZER_KEYS[key]
    if key in MEALPY_OPTIMIZER_REGISTRY:
        return MEALPY_OPTIMIZER_REGISTRY[key]
    if raw in _MEALPY_CLASS_NAMES:
        return raw
    matching = next((class_name for class_name in _MEALPY_CLASS_NAMES if class_name.upper() == key), None)
    if matching is not None:
        return matching
    raise ValueError(f"Optimizador no soportado o no instalado en MEALPY: {raw}")


def optimizer_acronym(name: str) -> str:
    resolved = resolve_optimizer_name(name)
    for acronym, class_name in MEALPY_OPTIMIZER_REGISTRY.items():
        if class_name == resolved:
            return acronym
    if resolved in CUSTOM_OPTIMIZERS:
        return resolved
    if resolved.startswith("Original") and len(resolved) > len("Original"):
        return resolved[len("Original"):].upper()
    return resolved


def list_available_optimizers() -> str:
    lines = []
    for acronym, class_name in MEALPY_OPTIMIZER_REGISTRY.items():
        lines.append(f"{acronym:<8} -> {class_name}")
    lines.append("")
    lines.append("Custom:")
    lines.extend(CUSTOM_OPTIMIZERS)
    return "\n".join(lines)
