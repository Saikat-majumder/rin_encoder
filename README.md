# rin-encodermap: Protein Residue Interaction Landscapes & PPI Distance Matrix Autoencoder

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning framework for learning continuous low-dimensional representations of protein conformational dynamics and protein-protein interaction (PPI) interfaces.

This codebase implements and extends the method introduced by:

> **Franke, L.; Peter, C.** *Visualizing the Residue Interaction Landscape of Proteins by Temporal Network Embedding.*  
> **J. Chem. Theory Comput.** 2023, 19 (10), 2985–2995.  
> [DOI: 10.1021/acs.jctc.2c01228](https://doi.org/10.1021/acs.jctc.2c01228)

---

## 🌟 Key Features

The repository supports two comprehensive pipelines:

1. **Original RIN Temporal Embedding**:
   - Converts molecular dynamics (MD) trajectories into dynamic Residue Interaction Networks (RINs).
   - Computes closeness centrality vectors for each residue across frames to produce high-dimensional fingerprint matrices.
   - Trains an EncoderMap autoencoder with a specialized Sketch-map sigmoid loss function that preserves multi-scale pairwise distances in low-dimensional latent space (e.g., 2D projections).

2. **High-Throughput PPI Distance Matrix & Cross-Validation**:
   - **Dual-Chain $C_\alpha$ PDB Parsing**: Automatically extracts coordinates for interacting protein chains (Chain A & Chain B).
   - **Probabilistic Distance Matrices (PDM)**: Computes inter-chain distance matrices normalized linearly between $5.0\text{ \AA}$ (full contact) and $80.0\text{ \AA}$ (non-interacting).
   - **Sliding-Window Feature Extraction**: Deconstructs interface contact maps into $128 \times 128$ windows with stride 64 ($50\%$ overlap), flattened into 16,384-dimensional feature vectors.
   - **Chunked Out-of-Core Caching**: Processes massive PDB collections in batches of 5,000 windows saved directly to disk (`pipeline_cache/`) with checkpointing (`.done` flags) to prevent memory crashes.
   - **Strict Disjoint Cross-Validation**: Deduplicates PDBs across 10 folds under a strict priority hierarchy (`Train > Valid > Test`) ensuring zero data leakage.
   - **Memory-Mapped (mmap) Training**: Streams multi-gigabyte `.npy` datasets directly from disk without exhausting system RAM.
   - **10-Fold Cross-Validation Engine**: Automated training loop featuring `tqdm` progress tracking, GPU acceleration, binarized contact accuracy, confusion matrix generation, and loss/accuracy trajectory plots.

---

## 📁 Repository Structure

```
rin_encoder/
├── README.md                      # Primary project documentation
├── README_1.md                    # Duplicate/alternative reference documentation
├── requirements.txt               # Python package dependencies
├── setup_venv.sh                  # Virtual environment bootstrap script
│
├── rin_encodermap/                # Core neural network & parsing package
│   ├── __init__.py
│   ├── encodermap_model.py        # EncoderMapNet architecture, sigmoid transforms, and loss functions
│   ├── pdb_parsing.py             # PDB parser, PDM distance matrices, and chunked dataset builder
│   ├── train_encodermap.py        # Standalone training script for 2D embedding
│   └── test_encodermap.py         # Evaluation and projection script for held-out frames
│
├── run_pipeline.py                # End-to-end dataset builder (parsing -> sliding windows -> cache merge)
├── combine_cache.py               # Memory-safe aggregator for cached chunk files
├── check_pdb.py                   # PDB integrity and residue count diagnostic utility
├── train_test_val.py              # Per-fold PDB deduplication generator (Train > Valid > Test)
├── create_fold_indices.py         # Maps PDB splits to binary sample index masks
├── run_cross_validation.py        # 10-fold cross-validation training & evaluation engine
├── testing_pipline.py             # Diagnostic test script for PDB chain extraction
│
├── folds/                         # Disjoint PDB identifier lists for 10 folds
│   ├── train_fold[1-10]_pdb.txt
│   ├── valid_fold[1-10]_pdb.txt
│   └── test_fold[1-10]_pdb.txt
│
├── fold_indices/                  # Precomputed boolean masks for memory-mapped dataset indexing
│   ├── train_mask_fold[1-10].npy
│   ├── valid_mask_fold[1-10].npy
│   └── test_mask_fold[1-10].npy
│
└── cross_val_results/             # Evaluation metrics, confusion matrices, and loss curves
    ├── confusion_matrix_fold[1-10].png
    ├── curves_fold[1-10].png
    └── test_predictions_fold[1-10].npy
```

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (strongly recommended for deep learning)

### Quick Setup

Clone the repository and run the setup script:

```bash
git clone https://github.com/Saikat-majumder/rin_encoder.git
cd rin_encoder

# Create and configure the virtual environment
bash setup_venv.sh
```

Or configure manually using `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies
| Library | Purpose |
|---|---|
| `torch` | Deep learning, GPU tensor operations, autoencoder training |
| `scipy` | Spatial distance computations (`cdist`) |
| `scikit-learn` | Classification metrics (Accuracy, Confusion Matrix) |
| `biopython` | Structural biological data handling |
| `networkx` | Residue Interaction Network graph algorithms |
| `seaborn` & `matplotlib` | Metric visualizations, confusion matrices, embedding plots |
| `tqdm` | Real-time CLI progress bars for folds, epochs, and batches |

---

## 🚀 Workflow & Execution Guide

### Pipeline A: High-Throughput PPI Distance Matrix & Cross-Validation

This pipeline processes complex dual-chain PDB structures, constructs sliding-window distance matrices, splits them into 10 disjoint cross-validation folds, and trains an EncoderMap representation model.

```
[ PDB Files ] 
      │
      ▼ (run_pipeline.py)
[ Sliding Windows (128x128) ] ──► [ Disk Cache (chunk_*.npy) ] ──► [ closeness_fingerprints.npy ]
                                                                             │
[ folds/train,valid,test_pdb.txt ]                                           │
      │                                                                      │
      ▼ (create_fold_indices.py)                                             │
[ fold_indices/*_mask_fold*.npy ] ───────────────────────────────────────────┤
                                                                             ▼ (run_cross_validation.py)
                                                                 [ 10-Fold CV Training & Evaluation ]
                                                                 - Confusion Matrices
                                                                 - Accuracy / Loss Curves
                                                                 - Predictions (.npy)
```

#### Step 1: Validate PDB Inputs
Inspect your PDB files to ensure valid atomic records and check residue count distributions:
```bash
python check_pdb.py
```

#### Step 2: Build the PDM Fingerprint Dataset
Run the one-click dataset generation script. This parses PDB chains, normalizes distances, creates $128 \times 128$ sliding windows (stride 64), caches them in chunks to disk to protect RAM, and aggregates them into `closeness_fingerprints.npy`:
```bash
python run_pipeline.py
```
*(If interrupted, rerun `python run_pipeline.py` or manually merge chunks using `python combine_cache.py`).*

#### Step 3: Generate Disjoint Folds
Enforce strict disjointness across all 10 folds (`Train > Valid > Test`):
```bash
python train_test_val.py
```
This generates cleaned PDB lists in `folds/` ensuring no test or validation structures leak into training sets.

#### Step 4: Map Folds to Sample Index Masks
Map each PDB's generated window count to global indices and save boolean index masks:
```bash
python create_fold_indices.py
```
The output masks will be written to `fold_indices/train_mask_fold{k}.npy`, `valid_mask_fold{k}.npy`, and `test_mask_fold{k}.npy`.

#### Step 5: Execute 10-Fold Cross-Validation
Train and evaluate the `EncoderMapNet` model across all 10 folds:
```bash
python run_cross_validation.py
```
**Features of the training engine**:
- **Zero-RAM Memory Mapping**: Uses `numpy.load(..., mmap_mode='r')` to stream multi-gigabyte datasets directly from disk.
- **Progress Tracking**: Real-time nested `tqdm` progress bars for Folds, Epochs, and Batches.
- **Metric Tracking**: Evaluates reconstruction MSE, contact accuracy (binarized at normalized threshold $< 0.1$, corresponding to $< 12.5\text{ \AA}$), and confusion matrices.
- **Automated Artifacts**: Saves confusion matrices (`confusion_matrix_fold{k}.png`), loss/accuracy curves (`curves_fold{k}.png`), and prediction tensors (`test_predictions_fold{k}.npy`) into `cross_val_results/`.

---

### Pipeline B: Single-Protein Trajectory Embedding (Franke & Peter, 2023)

For analyzing single MD trajectories or conformation ensembles using residue-level closeness centrality:

#### Step 1: Extract Residue Closeness Centrality Fingerprints
```python
from rin_encodermap.pdb_parsing import build_fingerprint_dataset
import numpy as np

# Parse MD trajectory frames and compute closeness centrality
X, file_list = build_fingerprint_dataset("path/to/trajectory_frames/*.pdb", cutoff=6.0)
np.save("closeness_fingerprints.npy", X)
```

#### Step 2: Train the 2D Latent Representation
```bash
python rin_encodermap/train_encodermap.py \
    --input closeness_fingerprints.npy \
    --output_dir runs/my_protein \
    --bottleneck_dim 2 \
    --n_steps 20000 \
    --batch_size 256
```
*(If `--input` is omitted, the script runs a synthetic 20-residue demo as a sanity check).*

#### Step 3: Project Held-Out Frames & Visualize the Landscape
```bash
python rin_encodermap/test_encodermap.py \
    --checkpoint runs/my_protein/encodermap_checkpoint.pt \
    --input closeness_fingerprints.npy \
    --indices runs/my_protein/val_indices.npy \
    --output_dir runs/my_protein
```
Outputs generated in `--output_dir`:
- `test_metrics.json`: Reconstruction MSE ($C_{auto}$) and Sketch loss ($C_{sketch}$).
- `test_frame_embedding.npy`: 2D latent coordinates for each evaluated frame.
- `test_embedding.png`: Visual scatter/hexbin plot of the residue interaction landscape.

---

## 📐 Mathematical Formulation

### 1. Distance Normalization
For $C_\alpha$ positions $r_A \in \text{Chain A}$ and $r_B \in \text{Chain B}$, pairwise Euclidean distances $d_{ij} = \|r_{A,i} - r_{B,j}\|_2$ are normalized into $[0, 1]$:
$$
\text{PDM}_{ij} = \text{clip}\left(\frac{d_{ij} - 5.0\text{ \AA}}{75.0\text{ \AA}}, 0.0, 1.0\right)
$$
- $d_{ij} \le 5.0\text{ \AA} \implies \text{PDM}_{ij} = 0.0$ (strong contact)
- $d_{ij} \ge 80.0\text{ \AA} \implies \text{PDM}_{ij} = 1.0$ (no interaction)

### 2. EncoderMap Loss Objective
The model is optimized using a composite objective:
$$
\mathcal{L} = C_{auto} + C_{sketch} + \lambda_{reg} C_{reg}
$$

#### Autoencoder Reconstruction Loss ($C_{auto}$)
Measures pixel-wise mean squared error between input window $x$ and reconstructed window $\hat{x}$:
$$
C_{auto} = \frac{1}{D} \sum_{k=1}^D (x_k - \hat{x}_k)^2
$$

#### Sketch-Map Distance Preservation Loss ($C_{sketch}$)
Preserves non-linear pairwise proximity relationships between sample pairs $(i, j)$ in a mini-batch:
$$
C_{sketch} = \frac{1}{B(B-1)} \sum_{i \neq j} \left[ F_H(D(x_i, x_j)) - F_L(d(z_i, z_j)) \right]^2
$$
where $D(x_i, x_j)$ is the high-dimensional Euclidean distance, $d(z_i, z_j)$ is the bottleneck latent distance, and $F(r)$ is the sigmoid transformation:
$$
F(r) = 1 - \left( 1 + (2^{a/b} - 1)\left(\frac{r}{\sigma}\right)^a \right)^{-b/a}
$$
The transformation suppresses noise from negligible distance fluctuations and disregards uninformative large distances that cannot be faithfully embedded.

#### Default Hyperparameters
- **Input Dimension**: $128 \times 128 = 16,384$
- **Bottleneck Dimension**: $32$ (for representation) or $2$ (for landscape visualization)
- **High-Dim Sigmoid $(\sigma_h, a_h, b_h)$**: $(0.5, 6.0, 6.0)$
- **Low-Dim Sigmoid $(\sigma_l, a_l, b_l)$**: $(1.0, 2.0, 6.0)$
- **Contact Threshold**: $\text{PDM} < 0.1 \implies \text{Distance} < 12.5\text{ \AA}$

---

## 📊 Summary of Results Artifacts

When running 10-fold cross-validation, artifacts are organized in `cross_val_results/`:

| Artifact | Format | Description |
|---|---|---|
| `confusion_matrix_fold{k}.png` | PNG image | Heatmap displaying true vs. predicted contact states on the test fold. |
| `curves_fold{k}.png` | PNG image | Dual-panel plot displaying training, validation, and test Loss and Accuracy per epoch. |
| `test_predictions_fold{k}.npy` | NumPy array | Reconstructed contact predictions for the held-out test fold. |

---

## 📖 Citation

If you use this codebase in your research, please cite the foundational work:

```bibtex
@article{Franke2023,
  author    = {Franke, Lennard and Peter, Christine},
  title     = {Visualizing the Residue Interaction Landscape of Proteins by Temporal Network Embedding},
  journal   = {Journal of Chemical Theory and Computation},
  volume    = {19},
  number    = {10},
  pages     = {2985--2995},
  year      = {2023},
  doi       = {10.1021/acs.jctc.2c01228},
  url       = {https://doi.org/10.1021/acs.jctc.2c01228}
}
```
