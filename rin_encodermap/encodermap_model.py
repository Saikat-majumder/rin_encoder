"""
encodermap_model.py
====================

Shared model, loss, and dataset definitions for the EncoderMap-style
dimensionality reduction of RIN closeness-centrality fingerprints
(Franke & Peter, JCTC 2023, https://doi.org/10.1021/acs.jctc.2c01228,
Section 4.3). Imported by both train_encodermap.py and test_encodermap.py
so the exact same architecture/loss is used in both places.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Sigmoid transform (Eq. 3)
# ----------------------------------------------------------------------
def sigmoid_transform(r, sigma, a, b, eps=1e-12):
    """
    SIG(r) = 1 - (1 + (2^(a/b) - 1) * (r/sigma)^a)^(-b/a)

    Suppresses the impact of small distances between very similar points
    and of very large distances that would be hard to reproduce accurately
    in low-dimensional space (paper, Eq. 3). Works on torch tensors
    (differentiable, used during training) as well as numpy arrays.
    """
    is_torch = isinstance(r, torch.Tensor)
    if is_torch:
        r = torch.clamp(r, min=eps)
    else:
        import numpy as np
        r = np.clip(r, eps, None)

    factor = (2.0 ** (a / b)) - 1.0
    inner = 1.0 + factor * (r / sigma) ** a
    return 1.0 - inner ** (-b / a)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class FingerprintDataset(Dataset):
    """Wraps an (N, D) array of closeness fingerprints."""

    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], idx


# ----------------------------------------------------------------------
# Autoencoder
# ----------------------------------------------------------------------
class EncoderMapNet(nn.Module):
    """
    Encoder: input_dim -> hidden_dims... -> bottleneck_dim (linear, no activation)
    Decoder: bottleneck_dim -> reversed(hidden_dims)... -> input_dim

    tanh hidden activations, matching the reference EncoderMap implementation.
    """

    def __init__(self, input_dim, bottleneck_dim=2, hidden_dims=(128, 128, 128)):
        super().__init__()

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        enc_layers += [nn.Linear(prev, bottleneck_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return code, recon


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def pairwise_euclidean(x):
    """Differentiable pairwise Euclidean distance matrix for a batch (B, D)."""
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-12)


def auto_cost(x, recon):
    """Reconstruction loss C_auto."""
    return nn.functional.mse_loss(recon, x)


def sketch_cost(x_batch, code_batch, sig_params_high, sig_params_low):
    """
    C_sketch (Eq. 2): mean squared difference between sigmoid-transformed
    pairwise distances in the high-dimensional input space (R_ij) and in
    the low-dimensional code space (r_ij), computed within a mini-batch.
    Fully differentiable w.r.t. the model's parameters through code_batch.
    """
    R = pairwise_euclidean(x_batch)
    r = pairwise_euclidean(code_batch)

    sigma_h, a_h, b_h = sig_params_high
    sigma_l, a_l, b_l = sig_params_low

    sig_high = sigmoid_transform(R, sigma_h, a_h, b_h)
    sig_low = sigmoid_transform(r, sigma_l, a_l, b_l)

    B = x_batch.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=x_batch.device)
    return ((sig_high[mask] - sig_low[mask]) ** 2).mean()


def l2_regularization(model):
    reg = 0.0
    for p in model.parameters():
        reg = reg + (p ** 2).sum()
    return reg


# ----------------------------------------------------------------------
# Checkpoint I/O (shared so train/test agree on the format)
# ----------------------------------------------------------------------
def save_checkpoint(model, path, bottleneck_dim, hidden_dims, sig_params_high, sig_params_low, input_dim):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "bottleneck_dim": bottleneck_dim,
            "hidden_dims": hidden_dims,
            "sig_params_high": sig_params_high,
            "sig_params_low": sig_params_low,
        },
        path,
    )


def load_checkpoint(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = EncoderMapNet(
        input_dim=ckpt["input_dim"],
        bottleneck_dim=ckpt["bottleneck_dim"],
        hidden_dims=ckpt["hidden_dims"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
