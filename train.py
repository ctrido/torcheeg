"""Train TorchEEG EEGNet on CHB-MIT seizure vs non-seizure windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torcheeg.models.cnn.eegnet import EEGNet

from src.build_dataset import COMMON_CHANNELS, make_torcheeg_dataset
from src.download_chbmit import download_subject


class WindowSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        y = int(y)
        return x, y


def split_indices(dataset) -> dict[str, list[int]]:
    info = dataset.info
    splits = {"train": [], "val": [], "test": []}
    for index, row in info.reset_index(drop=True).iterrows():
        splits[str(row["split"])].append(int(index))
    return splits


def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ys.append(y.numpy())
            probs.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "n_seizure": int(y_true.sum()),
        "n_nonseizure": int((1 - y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=["non-seizure", "seizure"], zero_division=0),
    }
    return metrics, y_true, y_prob, y_pred


def plot_curves(y_true, y_prob, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="non-seizure")
    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="seizure")
    axes[1].set_title("Predicted P(seizure)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non-seizure", "seizure"])
    ax.set_yticks([0, 1], ["non-seizure", "seizure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--io-path", type=Path, default=Path("data/processed/chb01_io"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_subject("chb01", args.data_dir)

    dataset, _, chunk_size = make_torcheeg_dataset(args.data_dir / "chb01", args.io_path)
    splits = split_indices(dataset)
    print({k: len(v) for k, v in splits.items()})

    train_ds = WindowSubset(dataset, splits["train"])
    val_ds = WindowSubset(dataset, splits["val"])
    test_ds = WindowSubset(dataset, splits["test"])

    y_train = [train_ds[i][1] for i in range(len(train_ds))]
    counts = np.bincount(y_train, minlength=2).astype(float)
    weights = [counts.sum() / (2 * counts[y] + 1e-6) for y in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    x0, _ = train_ds[0]
    if x0.ndim == 3:
        num_electrodes = int(x0.shape[1])
        time_points = int(x0.shape[2])
    else:
        num_electrodes = int(x0.shape[0])
        time_points = int(x0.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(
        chunk_size=time_points,
        num_electrodes=num_electrodes,
        dropout=0.5,
        kernel_1=64,
        kernel_2=16,
        F1=8,
        F2=16,
        D=2,
        num_classes=2,
    ).to(device)

    class_weights = torch.tensor([1.0, min(8.0, counts[0] / max(counts[1], 1.0))], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_state = None
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)
        print(row)
        score = val_metrics["balanced_accuracy"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_prob, _ = evaluate(model, test_loader, device)
    plot_curves(y_true, y_prob, args.out_dir / "roc_and_scores.png")
    plot_confusion(test_metrics["confusion_matrix"], args.out_dir / "confusion_matrix.png")
    torch.save(model.state_dict(), args.out_dir / "eegnet_chb01.pt")

    payload = {
        "subject": "chb01",
        "window_sec": 4.0,
        "channels": COMMON_CHANNELS,
        "num_electrodes": num_electrodes,
        "chunk_size": time_points,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "device": str(device),
        "history": history,
        "test": {k: v for k, v in test_metrics.items() if k != "report"},
        "classification_report": test_metrics["report"],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out_dir / "report.md").write_text(
        "\n".join(
            [
                "# CHB-MIT + TorchEEG EEGNet (pilot chb01)",
                "",
                f"- Windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}",
                f"- Test accuracy: {test_metrics['accuracy']:.3f}",
                f"- Test balanced accuracy: {test_metrics['balanced_accuracy']:.3f}",
                f"- Test F1: {test_metrics['f1']:.3f}",
                f"- Test ROC-AUC: {test_metrics['roc_auc']:.3f}" if test_metrics["roc_auc"] is not None else "- Test ROC-AUC: n/a",
                "",
                "```",
                test_metrics["report"].rstrip(),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(test_metrics["report"])
    print(f"Wrote results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
