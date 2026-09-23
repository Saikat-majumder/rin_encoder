"""
train_encodermap.py
====================

Train the EncoderMap autoencoder on closeness-centrality fingerprints
(one row per protein conformation / MD frame). Produces a checkpoint file
that test_encodermap.py can load to project new frames and evaluate
held-out data.

Usage:
    python train_encodermap.py --input closeness_fingerprints.npy \
        --output_dir runs/my_protein --n_steps 20000

If --input is omitted, a small synthetic demo dataset is generated so the
script can be run end-to-end without real data.

Fingerprints should already be scaled to [0, 1] (as in the paper, Section
4.2) -- see pdb_parsing.build_fingerprint_dataset for a helper that does
this scaling for you from a set of PDB frames.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from encodermap_model import (
    EncoderMapNet,
    FingerprintDataset,
    auto_cost,
    sketch_cost,
    l2_regularization,
    save_checkpoint,
)


def _evaluate_loss(model, loader, sig_params_high, sig_params_low, device):
    """Quick average total-loss estimate over a loader, no grad."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            code, recon = model(x_batch)
            c_auto = auto_cost(x_batch, recon)
            c_sketch = sketch_cost(x_batch, code, sig_params_high, sig_params_low)
            total += (c_auto + c_sketch).item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_encodermap(
    X_train,
    X_val=None,
    bottleneck_dim=2,
    hidden_dims=(128, 128, 128),
    batch_size=256,
    n_steps=20000,
    learning_rate=1e-5,
    reg_const=1e-5,
    sig_params_high=(1.0, 6.0, 6.0),   # (sigma_h, a_h, b_h)
    sig_params_low=(1.0, 2.0, 6.0),    # (sigma_l, a_l, b_l)
    device=None,
    checkpoint_path="encodermap_checkpoint.pt",
    log_every=200,
):
    """
    Train the EncoderMap autoencoder on high-dimensional closeness
    fingerprints X_train (n_samples, n_residues).

    Returns the trained model and a history dict of losses.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FingerprintDataset(X_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if X_val is not None:
        val_ds = FingerprintDataset(X_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    model = EncoderMapNet(
        input_dim=X_train.shape[1],
        bottleneck_dim=bottleneck_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"step": [], "train_total": [], "train_auto": [], "train_sketch": [], "val_total": []}

    step = 0
    t0 = time.time()
    model.train()

    while step < n_steps:
        for x_batch, _ in train_loader:
            if step >= n_steps:
                break
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            code, recon = model(x_batch)

            c_auto = auto_cost(x_batch, recon)
            c_sketch = sketch_cost(x_batch, code, sig_params_high, sig_params_low)
            c_reg = reg_const * l2_regularization(model)

            loss = c_auto + c_sketch + c_reg
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                val_loss = None
                if val_loader is not None:
                    val_loss = _evaluate_loss(model, val_loader, sig_params_high, sig_params_low, device)

                history["step"].append(step)
                history["train_total"].append(loss.item())
                history["train_auto"].append(c_auto.item())
                history["train_sketch"].append(c_sketch.item())
                history["val_total"].append(val_loss)

                elapsed = time.time() - t0
                msg = (f"step {step:6d}/{n_steps} | total {loss.item():.5f} "
                       f"| auto {c_auto.item():.5f} | sketch {c_sketch.item():.5f} "
                       f"| {elapsed:.1f}s")
                if val_loss is not None:
                    msg += f" | val {val_loss:.5f}"
                print(msg)

            step += 1

    save_checkpoint(
        model, checkpoint_path,
        bottleneck_dim=bottleneck_dim,
        hidden_dims=hidden_dims,
        sig_params_high=sig_params_high,
        sig_params_low=sig_params_low,
        input_dim=X_train.shape[1],
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    return model, history


def plot_training_curves(history, save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history["step"], history["train_total"], label="train total")
    plt.plot(history["step"], history["train_auto"], label="train auto (reconstruction)")
    plt.plot(history["step"], history["train_sketch"], label="train sketch (distance)")
    if any(v is not None for v in history["val_total"]):
        plt.plot(history["step"], history["val_total"], label="val total", linestyle="--")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title("EncoderMap training curves")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train an EncoderMap embedding of RIN closeness fingerprints.")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a .npy file (n_frames, n_residues) of closeness fingerprints. "
                              "If omitted, a synthetic demo dataset is used.")
    parser.add_argument("--bottleneck_dim", type=int, default=2)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--reg_const", type=float, default=1e-5)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="Fraction of frames held out for validation during training.")
    parser.add_argument("--sig_params_high", type=float, nargs=3, default=[1.0, 6.0, 6.0],
                         help="sigma_h a_h b_h for the high-dimensional sigmoid transform")
    parser.add_argument("--sig_params_low", type=float, nargs=3, default=[1.0, 2.0, 6.0],
                         help="sigma_l a_l b_l for the low-dimensional sigmoid transform")
    parser.add_argument("--output_dir", type=str, default="runs/default")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input is not None:
        X = np.load(args.input)
    else:
        print("No --input given: generating a synthetic demo dataset (2000 frames, 20 residues).")
        rng = np.random.default_rng(0)
        n_frames, n_residues = 2000, 20
        base_a = rng.uniform(0.2, 0.4, size=n_residues)
        base_b = rng.uniform(0.6, 0.9, size=n_residues)
        labels = rng.integers(0, 2, size=n_frames)
        X = np.where(labels[:, None] == 0, base_a, base_b) + rng.normal(0, 0.03, size=(n_frames, n_residues))
        X = np.clip(X, 0, 1)

    # simple random train/val split (frames are treated as i.i.d. samples
    # for this feature-based embedding, as in the paper)
    n = X.shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val = X[train_idx], X[val_idx]

    print(f"Training on {X_train.shape[0]} frames, validating on {X_val.shape[0]} frames "
          f"({X.shape[1]}-dimensional closeness fingerprints).")

    model, history = train_encodermap(
        X_train,
        X_val=X_val,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dims=tuple(args.hidden_dims),
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        reg_const=args.reg_const,
        sig_params_high=tuple(args.sig_params_high),
        sig_params_low=tuple(args.sig_params_low),
        checkpoint_path=os.path.join(args.output_dir, "encodermap_checkpoint.pt"),
    )

    plot_training_curves(history, save_path=os.path.join(args.output_dir, "training_curves.png"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # keep the val split around so test_encodermap.py can be pointed at the
    # exact same held-out frames if desired
    np.save(os.path.join(args.output_dir, "val_indices.npy"), val_idx)
    np.save(os.path.join(args.output_dir, "train_indices.npy"), train_idx)

    print(f"Done. Checkpoint and logs written to {args.output_dir}/")


if __name__ == "__main__":
    main()
