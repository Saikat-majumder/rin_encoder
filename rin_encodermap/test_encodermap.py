"""
test_encodermap.py
===================

Load a checkpoint produced by train_encodermap.py, project held-out (or
any new) closeness fingerprints into the learned low-dimensional map, and
report reconstruction / distance-preservation error on that data.

Usage:
    python test_encodermap.py \
        --checkpoint runs/my_protein/encodermap_checkpoint.pt \
        --input closeness_fingerprints.npy \
        --indices runs/my_protein/val_indices.npy \
        --output_dir runs/my_protein

If --indices is omitted, --input is assumed to already be just the
test/held-out set. If both --checkpoint and --input are omitted, the
script regenerates the same synthetic demo dataset train_encodermap.py
produces by default, trains nothing, and simply demonstrates loading +
projecting + evaluating (useful as a smoke test).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encodermap_model import FingerprintDataset, sketch_cost, load_checkpoint


def project(model, X, device=None, batch_size=4096):
    """Project fingerprints X (N, D) into the learned low-dim map.
    Returns (N, bottleneck_dim)."""
    device = device or next(model.parameters()).device
    model.eval()
    ds = FingerprintDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    codes = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            code, _ = model(x_batch)
            codes.append(code.cpu().numpy())
    return np.concatenate(codes, axis=0)


def test_encodermap(model, X_test, sig_params_high, sig_params_low, device=None, batch_size=4096):
    """
    Evaluate a trained model on held-out data.

    Reports:
      - mean/std reconstruction error (C_auto) per frame on the test set
      - mean sketch (distance-preservation) error, estimated on random
        mini-batches drawn from the test set (computing full pairwise
        distances over an entire held-out set is usually too large to do
        at once)
      - the low-dimensional embedding of every test frame

    Returns (results_dict, codes, recon_error_per_frame).
    """
    device = device or next(model.parameters()).device
    model.eval()

    X_t = torch.as_tensor(X_test, dtype=torch.float32).to(device)

    recon_errors = []
    codes_list = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            chunk = X_t[i:i + batch_size]
            code, recon = model(chunk)
            recon_errors.append(
                nn.functional.mse_loss(recon, chunk, reduction="none").mean(dim=1).cpu().numpy()
            )
            codes_list.append(code.cpu().numpy())
    recon_error_per_frame = np.concatenate(recon_errors)
    codes = np.concatenate(codes_list, axis=0)

    n_eval_batches = max(1, X_t.shape[0] // batch_size)
    sketch_errors = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(n_eval_batches):
            idx = rng.choice(X_t.shape[0], size=min(batch_size, X_t.shape[0]), replace=False)
            batch = X_t[idx]
            code, _ = model(batch)
            sketch_errors.append(sketch_cost(batch, code, sig_params_high, sig_params_low).item())

    results = {
        "n_test_frames": int(X_test.shape[0]),
        "mean_reconstruction_mse": float(recon_error_per_frame.mean()),
        "std_reconstruction_mse": float(recon_error_per_frame.std()),
        "mean_sketch_error": float(np.mean(sketch_errors)),
    }
    return results, codes, recon_error_per_frame


def plot_embedding(codes, color_by=None, title="Residue interaction landscape", cbar_label="value",
                    save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 7))
    if color_by is not None:
        sc = plt.scatter(codes[:, 0], codes[:, 1], c=color_by, cmap="viridis", s=3, alpha=0.6)
        plt.colorbar(sc, label=cbar_label)
    else:
        plt.hexbin(codes[:, 0], codes[:, 1], gridsize=200, cmap="viridis", bins="log")
        plt.colorbar(label="log(density)")

    plt.xlabel("EncoderMap dimension 1")
    plt.ylabel("EncoderMap dimension 2")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test/evaluate a trained EncoderMap checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a checkpoint .pt file from train_encodermap.py")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a .npy fingerprint array to project/evaluate. "
                              "If --indices is also given, this should be the FULL dataset "
                              "used for training (indices are applied to it).")
    parser.add_argument("--indices", type=str, default=None,
                         help="Optional .npy file of row indices into --input selecting the test set "
                              "(e.g. runs/.../val_indices.npy from train_encodermap.py).")
    parser.add_argument("--output_dir", type=str, default="runs/default")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None or args.input is None:
        print("No --checkpoint/--input given: running an end-to-end smoke test "
              "(trains a tiny model on synthetic data, then tests it).")
        from train_encodermap import train_encodermap

        rng = np.random.default_rng(0)
        n_frames, n_residues = 2000, 20
        base_a = rng.uniform(0.2, 0.4, size=n_residues)
        base_b = rng.uniform(0.6, 0.9, size=n_residues)
        labels = rng.integers(0, 2, size=n_frames)
        X = np.where(labels[:, None] == 0, base_a, base_b) + rng.normal(0, 0.03, size=(n_frames, n_residues))
        X = np.clip(X, 0, 1)

        perm = rng.permutation(n_frames)
        n_test = n_frames // 10
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        X_train, X_test = X[train_idx], X[test_idx]

        ckpt_path = os.path.join(args.output_dir, "smoke_test_checkpoint.pt")
        model, _ = train_encodermap(X_train, n_steps=300, batch_size=128, checkpoint_path=ckpt_path)
        sig_params_high, sig_params_low = (1.0, 6.0, 6.0), (1.0, 2.0, 6.0)
    else:
        model, ckpt = load_checkpoint(args.checkpoint, device=device)
        sig_params_high = tuple(ckpt["sig_params_high"])
        sig_params_low = tuple(ckpt["sig_params_low"])

        X = np.load(args.input)
        if args.indices is not None:
            idx = np.load(args.indices)
            X_test = X[idx]
        else:
            X_test = X

    results, codes, recon_err = test_encodermap(model, X_test, sig_params_high, sig_params_low, device=device)
    print("Test-set evaluation:")
    print(json.dumps(results, indent=2))

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(args.output_dir, "test_frame_embedding.npy"), codes)

    plot_embedding(
        codes,
        color_by=recon_err,
        title="Held-out test frames, colored by reconstruction error",
        cbar_label="reconstruction MSE",
        save_path=os.path.join(args.output_dir, "test_embedding.png"),
    )

    print(f"Done. Test metrics, embedding, and plot written to {args.output_dir}/")


if __name__ == "__main__":
    main()
