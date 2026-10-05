#!/usr/bin/env python3
"""
filter_dataset.py

Filters closeness_fingerprints.npy to keep only windows with ≥MIN_CONTACT_FRACTION
contact pixels (distance < CONTACT_THRESHOLD). Saves filtered data and updated fold masks.

This addresses extreme class imbalance (99% non-contact pixels) that causes the model
to predict everything as non-contact.
"""

import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "closeness_fingerprints.npy"
OUTPUT_FILE = "closeness_fingerprints_filtered.npy"
INDICES_DIR = "fold_indices"
OUTPUT_INDICES_DIR = "fold_indices_filtered"

CONTACT_THRESHOLD = 0.067  # 10Å normalized distance
MIN_CONTACT_FRACTION = 0.05  # Keep windows with ≥5% contact pixels
WINDOW_SIZE = 128
CHUNK_SIZE = 10000  # Process in chunks to manage memory

def main():
    print("="*70)
    print("FILTERING DATASET FOR CLASS BALANCE")
    print("="*70)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Contact threshold: {CONTACT_THRESHOLD} (≤10Å)")
    print(f"Minimum contact fraction: {MIN_CONTACT_FRACTION} (≥5%)")
    print("-"*70)

    # 1. Load data with mmap for memory efficiency
    print("Loading dataset (mmap)...")
    X_full = np.load(INPUT_FILE, mmap_mode='r')
    n_samples, n_features = X_full.shape
    print(f"  Original shape: {X_full.shape}")
    print(f"  Original size: {n_samples:,} windows")

    # 2. Find which windows meet the contact fraction threshold
    print(f"\nScanning for windows with ≥{MIN_CONTACT_FRACTION*100}% contact pixels...")
    keep_mask = np.zeros(n_samples, dtype=bool)

    # Process in chunks to avoid memory issues
    for start_idx in tqdm(range(0, n_samples, CHUNK_SIZE), desc="  Scanning chunks"):
        end_idx = min(start_idx + CHUNK_SIZE, n_samples)
        chunk = X_full[start_idx:end_idx]

        # Calculate contact fraction per window
        contact_fractions = (chunk < CONTACT_THRESHOLD).mean(axis=1)
        keep_mask[start_idx:end_idx] = contact_fractions >= MIN_CONTACT_FRACTION

    n_keep = keep_mask.sum()
    print(f"  Windows kept: {n_keep:,} / {n_samples:,} ({n_keep/n_samples*100:.2f}%)")
    print(f"  Windows removed: {n_samples - n_keep:,} ({(1 - n_keep/n_samples)*100:.2f}%)")

    # 3. Extract filtered windows
    print("\nExtracting filtered windows...")
    keep_indices = np.where(keep_mask)[0]

    # Process in chunks and save
    X_filtered_parts = []
    for i in tqdm(range(0, len(keep_indices), CHUNK_SIZE), desc="  Extracting chunks"):
        chunk_indices = keep_indices[i:i+CHUNK_SIZE]
        X_filtered_parts.append(X_full[chunk_indices])

    X_filtered = np.concatenate(X_filtered_parts, axis=0)
    print(f"  Filtered shape: {X_filtered.shape}")

    # 4. Save filtered dataset
    print(f"\nSaving to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, X_filtered)
    print(f"  Saved {X_filtered.shape[0]:,} windows ({X_filtered.nbytes / 1e9:.2f} GB)")

    # 5. Update fold masks to match filtered indices
    print(f"\nUpdating fold masks...")
    os.makedirs(OUTPUT_INDICES_DIR, exist_ok=True)

    # Create mapping from old indices to new indices
    old_to_new = np.full(n_samples, -1, dtype=np.int32)
    old_to_new[keep_indices] = np.arange(len(keep_indices))

    for k in tqdm(range(1, 11), desc="  Processing folds"):
        for split in ['train', 'valid', 'test']:
            old_mask_path = os.path.join(INDICES_DIR, f"{split}_mask_fold{k}.npy")
            new_mask_path = os.path.join(OUTPUT_INDICES_DIR, f"{split}_mask_fold{k}.npy")

            if not os.path.exists(old_mask_path):
                print(f"    ⚠️  Missing {old_mask_path}, skipping")
                continue

            # Load old mask
            old_mask = np.load(old_mask_path)

            # Find which kept windows were in this fold
            # old_mask[i] = True means sample i was in this fold
            # We keep it if both: old_mask[i] AND keep_mask[i]
            old_indices_in_fold = np.where(old_mask)[0]
            kept_from_fold = np.intersect1d(old_indices_in_fold, keep_indices)

            # Map to new indices
            new_indices_in_fold = old_to_new[kept_from_fold]

            # Create new mask
            new_mask = np.zeros(len(X_filtered), dtype=bool)
            new_mask[new_indices_in_fold] = True

            # Save
            np.save(new_mask_path, new_mask)

    print(f"  Saved updated masks to {OUTPUT_INDICES_DIR}/")

    # 6. Report final statistics per fold
    print("\n" + "="*70)
    print("FOLD STATISTICS (FILTERED)")
    print("="*70)

    for k in range(1, 11):
        train_mask = np.load(os.path.join(OUTPUT_INDICES_DIR, f"train_mask_fold{k}.npy"))
        valid_mask = np.load(os.path.join(OUTPUT_INDICES_DIR, f"valid_mask_fold{k}.npy"))
        test_mask = np.load(os.path.join(OUTPUT_INDICES_DIR, f"test_mask_fold{k}.npy"))

        n_train = train_mask.sum()
        n_valid = valid_mask.sum()
        n_test = test_mask.sum()

        print(f"Fold {k:2d}: Train={n_train:6,} | Valid={n_valid:5,} | Test={n_test:5,}")

    print("\n" + "="*70)
    print("FILTERING COMPLETE")
    print("="*70)
    print(f"✅ Filtered dataset: {OUTPUT_FILE}")
    print(f"✅ Updated masks: {OUTPUT_INDICES_DIR}/")
    print(f"\nNext step: Update run_cross_validation.py to use:")
    print(f"  NPY_FILE = '{OUTPUT_FILE}'")
    print(f"  INDICES_DIR = '{OUTPUT_INDICES_DIR}'")

if __name__ == "__main__":
    main()
