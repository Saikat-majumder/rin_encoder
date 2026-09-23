# rin-encodermap

Visualize the residue interaction landscape of a protein by combining
Residue Interaction Networks (RINs), closeness centrality, and an
EncoderMap-style autoencoder embedding.

Reimplements the method of:

> Franke, L.; Peter, C. *Visualizing the Residue Interaction Landscape of
> Proteins by Temporal Network Embedding.* J. Chem. Theory Comput. 2023,
> 19, 2985–2995. https://doi.org/10.1021/acs.jctc.2c01228

## Pipeline

1. **`rin_encodermap/pdb_parsing.py`** — parse PDB frames, build a Residue
   Interaction Network (contact graph, 6 Å cutoff, sequence-neighbor
   exclusion), and compute each residue's closeness centrality. Stacking
   the N-dimensional closeness vector over all simulation frames gives an
   `(n_frames, n_residues)` fingerprint matrix, scaled to `[0, 1]`.
2. **`rin_encodermap/encodermap_model.py`** — the autoencoder
   architecture and the Sketch-map-style sigmoid pairwise-distance loss
   (Eq. 2–3 of the paper), shared by training and testing.
3. **`rin_encodermap/train_encodermap.py`** — trains the autoencoder on a
   fingerprint matrix and saves a checkpoint.
4. **`rin_encodermap/test_encodermap.py`** — loads a checkpoint, projects
   held-out fingerprints into the learned 2D map, and reports
   reconstruction / distance-preservation error.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### 1. Build fingerprints from PDB frames

```python
from rin_encodermap.pdb_parsing import build_fingerprint_dataset
import numpy as np

X, files = build_fingerprint_dataset("frames/frame_*.pdb", cutoff=6.0)
np.save("closeness_fingerprints.npy", X)
```

Or, for a single structure:

```bash
python rin_encodermap/pdb_parsing.py my_protein.pdb
```

### 2. Train

```bash
python rin_encodermap/train_encodermap.py \
    --input closeness_fingerprints.npy \
    --output_dir runs/my_protein \
    --n_steps 20000
```

Runs with no `--input` on a small synthetic demo dataset, useful as a
sanity check.

### 3. Test / project new or held-out frames

```bash
python rin_encodermap/test_encodermap.py \
    --checkpoint runs/my_protein/encodermap_checkpoint.pt \
    --input closeness_fingerprints.npy \
    --indices runs/my_protein/val_indices.npy \
    --output_dir runs/my_protein
```

This writes `test_metrics.json`, `test_frame_embedding.npy`, and
`test_embedding.png` to `--output_dir`.

## Notes

- Sigmoid parameters `(sigma, a, b)` default to the paper's Trp-Cage
  values (Table 1): high-dim `(1.0, 6.0, 6.0)`, low-dim `(1.0, 2.0, 6.0)`.
  Adjust with `--sig_params_high` / `--sig_params_low` for other proteins.
- The paper notes that qualitative results are fairly robust to these
  parameters, but they should be chosen by visual inspection of the
  resulting map for a given system.
