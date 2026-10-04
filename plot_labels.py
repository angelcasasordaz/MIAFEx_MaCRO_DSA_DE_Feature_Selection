"""Display labels and proposed-method selection; never scientific identities."""
import re


def method_key(name):
    """Normalize display aliases and TF suffixes without changing stored keys."""
    key = re.sub(r"_(?:VSTF|SSTF)_\d+$", "", str(name).strip().upper())
    if key.startswith("ORIGINAL"):
        key = key[len("ORIGINAL"):]
    return "DSADE" if key in {"DSA-DE", "DSA_DE"} else key


def plot_display_label(name, present_methods=()):
    # MIAFEx retains MaCRO-DE-t and MaCRO-DE-t-v2. The Code Smell DSA-DE
    # manuscript alias is intentionally inapplicable to this project.
    text = str(name)
    base = re.sub(r"_(?:VSTF|SSTF)_\d+$", "", text, flags=re.IGNORECASE)
    label = {"MACRO-DE": "MaCRO-DE", "MACRO-DE-T": "MaCRO-DE-t",
             "MACRO-DE-T-V2": "MaCRO-DE-t-v2"}.get(method_key(base), base)
    return label + text[len(base):]


def report_display_label(label, present_labels=()):
    method, separator, suffix = str(label).partition(" | ")
    return plot_display_label(method, present_labels) + separator + suffix


def primary_algorithm(algorithms, args=None):
    """First active proposed family in the current configuration, else reference.

    Compare base methods so transfer variants share highlighting. Return an
    existing complete identity for pairwise comparisons; no algorithm is renamed.
    """
    from reporting.core import framework
    configured = getattr(args, "optimizers", None) or framework().OPTIMIZERS
    families = {"MACRO-DE", "MACRO-DE-T", "MACRO-DE-T-V2", "DSADE"}
    for method in configured:
        if method_key(method) in families:
            for algorithm in algorithms:
                if method_key(algorithm) == method_key(method):
                    return algorithm
    return algorithms[0] if algorithms else None


def is_primary(algorithm, reference):
    return reference is not None and method_key(algorithm) == method_key(reference)
