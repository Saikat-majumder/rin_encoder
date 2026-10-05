#!/usr/bin/env python3
"""
create_filtered_folds.py

Re-splits the filtered dataset (closeness_fingerprints_filtered.npy) into proper
10-fold cross-validation with balanced train/valid/test splits.

Since the original test sets had too few contact-rich windows, we re-partition
the filtered data directly using sklearn's KFold.
"""

import os
import numpy as np
from sklearn.model_selection import KFold

# --- CONFIGURATION ---
FILTERED_FILE = "closeness_fingerprints_filtered.npy"
OUTPUT_DIR = "fold_indices_filtered"
N_FOLDS = 10
RANDOM_SEED = 42

def main():
    print("="*70)
    print("RE-SPLITTING FILTERED DATASET INTO 10-FOLD CV")
    print("="*70)
    print(f"Input: {FILTERED_FILE}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Folds: {N_FOLDS}")
    print("-"*70)

    # 1. Load filtered data (just to get sample count)
    print("Loading filtered dataset (mmap)...")
    X = np.load(FILTERED_FILE, mmap_mode='r')
    n_samples = X.shape[0]
    print(f"  Shape: {X.shape}")
    print(f"  Samples: {n_samples:,}")

    # 2. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Generate 10-fold splits
    print(f"\nGenerating {N_FOLDS}-fold splits...")
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_indices = np.arange(n_samples)

    print("\n" + "="*70)
    print("FOLD SPLITS")
    print("="*70)

    for fold_idx, (train_valid_idx, test_idx) in enumerate(kfold.split(all_indices), start=1):
        # Further split train_valid into train and valid (90/10 split of the non-test data)
        n_train_valid = len(train_valid_idx)
        n_valid = max(1, int(n_train_valid * 0.1))  # 10% for validation, at least 1
        n_train = n_train_valid - n_valid

        # Shuffle train_valid and split
        np.random.seed(RANDOM_SEED + fold_idx)
        shuffled = np.random.permutation(train_valid_idx)
        train_idx = shuffled[:n_train]
        valid_idx = shuffled[n_train:]

        # Create boolean masks
        train_mask = np.zeros(n_samples, dtype=bool)
        valid_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)

        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

        # Save masks
        np.save(os.path.join(OUTPUT_DIR, f"train_mask_fold{fold_idx}.npy"), train_mask)
        np.save(os.path.join(OUTPUT_DIR, f"valid_mask_fold{fold_idx}.npy"), valid_mask)
        np.save(os.path.join(OUTPUT_DIR, f"test_mask_fold{fold_idx}.npy"), test_mask)

        # Verify disjointness
        overlap_tv = np.sum(train_mask & valid_mask)
        overlap_tt = np.sum(train_mask & test_mask)
        overlap_vt = np.sum(valid_mask & test_mask)
        total_overlap = overlap_tv + overlap_tt + overlap_vt

        status = "✅" if total_overlap == 0 else "❌"
        print(f"{status} Fold {fold_idx:2d}: Train={train_mask.sum():6,} | "
              f"Valid={valid_mask.sum():5,} | Test={test_mask.sum():5,} | "
              f"Overlaps={total_overlap}")

    print("\n" + "="*70)
    print("FOLD CREATION COMPLETE")
    print("="*70)
    print(f"✅ Masks saved to {OUTPUT_DIR}/")
    print(f"\nNext: Run cross-validation with:")
    print(f"  NPY_FILE = '{FILTERED_FILE}'")
    print(f"  INDICES_DIR = '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
