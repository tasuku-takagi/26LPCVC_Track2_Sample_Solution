import argparse
import json
import os

import h5py
import numpy as np
import torch


def topk_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[torch.Tensor]:
    """
    Compute top-k classification accuracy.

    Args:
        preds:   Logits or probabilities of shape (N, num_classes).
        targets: Ground-truth class indices of shape (N,).
        topk:    Tuple of k values to evaluate (e.g. (1, 5)).

    Returns:
        List of accuracy percentages, one per k value.
    """
    maxk = max(topk)
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [
        (correct[:k].reshape(-1).float().sum(0) / preds.size(0)) * 100.0
        for k in topk
    ]


def load_logits(h5_path: str) -> np.ndarray:
    """Load and stack inference logits from an HDF5 file produced by run_inference.py."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"H5 file not found: '{h5_path}'. "
            "Run run_inference.py first to generate it."
        )
    logits = []
    with h5py.File(h5_path, "r") as f:
        grp = f["data/0"]
        sorted_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))
        for k in sorted_keys:
            logits.append(grp[k][...].squeeze())
    return np.stack(logits, axis=0)


def load_labels(manifest_path: str, class_to_idx: dict) -> list[int]:
    """Read ground-truth labels from the manifest written by preprocess_and_save.py."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: '{manifest_path}'. "
            "Run preprocess_and_save.py first."
        )
    labels = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            label = record["label"]
            if label not in class_to_idx:
                raise KeyError(
                    f"Label '{label}' from manifest not found in class map. "
                    "Check that the correct class_map.json is being used."
                )
            labels.append(class_to_idx[label])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate QAI Hub inference results against ground-truth labels."
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to the HDF5 logits file produced by run_inference.py.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Path to the manifest.jsonl file produced by preprocess_and_save.py. "
             "Defaults to <OUT_ROOT>/manifest.jsonl inside the preprocessed directory.",
    )
    parser.add_argument(
        "--class_map",
        default="class_map.json",
        help="Path to the class_map.json file (default: class_map.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample predictions alongside the final accuracy.",
    )
    args = parser.parse_args()

    # --- Load class map ---
    if not os.path.exists(args.class_map):
        raise FileNotFoundError(
            f"Class map not found: '{args.class_map}'. "
            "Provide the correct path with --class_map."
        )
    with open(args.class_map, "r", encoding="utf-8") as f:
        class_to_idx: dict = json.load(f)
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    # --- Resolve manifest path ---
    manifest_path = args.manifest if args.manifest else ""
    if not manifest_path:
        raise ValueError(
            "'--manifest' is required. "
            "Point it to the manifest.jsonl file produced by preprocess_and_save.py."
        )

    # --- Load data ---
    raw_logits = load_logits(args.h5)
    labels = load_labels(manifest_path, class_to_idx)

    logits = torch.as_tensor(raw_logits, dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    # --- Handle size mismatch (partial inference results) ---
    n_logits = probs.shape[0]
    n_labels = label_tensor.shape[0]
    if n_labels > n_logits:
        print(
            f"ℹ️  H5 has {n_logits} results but manifest has {n_labels} labels — "
            f"truncating labels to the first {n_logits} for partial evaluation."
        )
        label_tensor = label_tensor[:n_logits]
    elif n_logits > n_labels:
        raise ValueError(
            f"H5 has more results ({n_logits}) than manifest labels ({n_labels}). "
            "Ensure the H5 file and manifest match the same dataset."
        )

    # --- Optional per-sample preview ---
    if args.verbose:
        pred_indices = torch.argmax(probs, dim=1)
        n_show = min(10, pred_indices.shape[0])
        print(f"\nFirst {n_show} samples:")
        for i in range(n_show):
            p = pred_indices[i].item()
            g = label_tensor[i].item()
            correct = "✓" if p == g else "✗"
            print(f"  [{i}] {correct}  pred={idx_to_class[p]} ({p})  gt={idx_to_class[g]} ({g})")

    # --- Accuracy ---
    acc1, acc5 = topk_accuracy(probs, label_tensor, topk=(1, 5))
    print(f"\nTop-1 Accuracy: {acc1.item():.2f}%")
    print(f"Top-5 Accuracy: {acc5.item():.2f}%")


if __name__ == "__main__":
    main()
