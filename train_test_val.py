import os

# Define the exact directory where your original split files are located
FOLDS_DIR = "/home/subhadip/rin_encoder_1/folds"

# Define the NEW directory where the cleaned PDB lists will be saved
OUTPUT_DIR = "/home/subhadip/rin_encoder/folds"

def extract_pdbs(filepath):
    """Extracts unique PDB IDs from a split file."""
    pdbs = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if not parts: continue
            img_path = parts[0]
            filename = os.path.basename(img_path)
            pdb_id = filename.split('_')[0]
            pdbs.add(f"{pdb_id}.pdb")
    return pdbs

def save_pdbs(filepath, pdbs):
    """Saves a set of PDB IDs to a text file."""
    with open(filepath, 'w') as f:
        for pdb in sorted(list(pdbs)):
            f.write(f"{pdb}\n")

def main():
    print("🚀 Starting Per-Fold PDB Deduplication...")
    print(f"📂 Source directory: {FOLDS_DIR}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print("️  Priority: Train > Valid > Test (Resetting pool for each fold)")
    print("-" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each fold independently
    for k in range(1, 11):
        train_file = os.path.join(FOLDS_DIR, f"train_fold{k}.txt")
        valid_file = os.path.join(FOLDS_DIR, f"valid_fold{k}.txt")
        test_file = os.path.join(FOLDS_DIR, "test.txt") # Test file is the same source for all folds
        
        if not os.path.exists(train_file) or not os.path.exists(valid_file) or not os.path.exists(test_file):
            print(f"⚠️  Skipping Fold {k} (Missing files)")
            continue
            
        # 1. Extract raw PDBs for this specific fold
        train_raw = extract_pdbs(train_file)
        valid_raw = extract_pdbs(valid_file)
        test_raw = extract_pdbs(test_file)
        
        # 2. Apply Priority: Train (Top Priority)
        train_clean = train_raw
        
        # 3. Apply Priority: Valid (Medium Priority) - loses overlaps with Train
        valid_clean = valid_raw - train_clean
        
        # 4. Apply Priority: Test (Lowest Priority) - loses overlaps with Train AND Valid
        test_clean = test_raw - train_clean - valid_clean
        
        # 5. Save the clean, disjoint sets for this fold
        save_pdbs(os.path.join(OUTPUT_DIR, f"train_fold{k}_pdb.txt"), train_clean)
        save_pdbs(os.path.join(OUTPUT_DIR, f"valid_fold{k}_pdb.txt"), valid_clean)
        save_pdbs(os.path.join(OUTPUT_DIR, f"test_fold{k}_pdb.txt"), test_clean)
        
        # 6. Verify strict disjointness
        overlap_tv = len(train_clean & valid_clean)
        overlap_tt = len(train_clean & test_clean)
        overlap_vt = len(valid_clean & test_clean)
        total_overlaps = overlap_tv + overlap_tt + overlap_vt
        
        status = "✅" if total_overlaps == 0 else "❌"
        print(f"{status} Fold {k:>2} | Train: {len(train_clean):>5} | Valid: {len(valid_clean):>5} | Test: {len(test_clean):>5} | Overlaps: {total_overlaps}")

    print("-" * 70)
    print("🎉 Per-fold deduplication complete!")
    print("💡 For every fold k, the Train, Valid, and Test sets are 100% disjoint.")
    print("   (Note: Test set PDBs are saved as test_fold{k}_pdb.txt to prevent overwriting).")

if __name__ == "__main__":
    main()