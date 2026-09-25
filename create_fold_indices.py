import os
import glob
import numpy as np

# --- EXACT CONFIGURATION PROVIDED ---
PDB_DIR = "/home/subhadip/rin_encoder_1/PDB-Uniprot"
FOLDS_DIR = "/home/subhadip/rin_encoder/folds"
OUTPUT_DIR = "fold_indices"

WINDOW_SIZE = 128
STRIDE = 64

def get_pdb_window_count(filepath):
    """Calculates how many sliding windows a PDB file generates."""
    chains = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain_id = line[21]
                if chain_id not in chains: chains[chain_id] = 0
                chains[chain_id] += 1
    
    sorted_keys = sorted(chains.keys())
    if len(sorted_keys) < 2: return 0
    
    lenA, lenB = chains[sorted_keys[0]], chains[sorted_keys[1]]
    if lenA < WINDOW_SIZE or lenB < WINDOW_SIZE: return 0
    
    windows_A = max(0, (lenA - WINDOW_SIZE) // STRIDE + 1)
    windows_B = max(0, (lenB - WINDOW_SIZE) // STRIDE + 1)
    return windows_A * windows_B

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pdb_pattern = os.path.join(PDB_DIR, "*.pdb")
    pdb_files = sorted(glob.glob(pdb_pattern))
    
    print(f"🔍 Scanning PDBs in: {PDB_DIR}")
    print(f"🔍 Scanning folds in: {FOLDS_DIR}")
    print("-" * 70)
    
    pdb_to_indices = {}
    current_idx = 0
    
    for pdb_file in pdb_files:
        filename = os.path.basename(pdb_file)
        
        # BULLETPROOF FIX: Remove ALL '.pdb' strings, split by '_', take first, add exactly one '.pdb'
        clean_name = filename.replace('.pdb', '') 
        pdb_id = clean_name.split('_')[0] + ".pdb"
        
        count = get_pdb_window_count(pdb_file)
        if count > 0:
            pdb_to_indices[pdb_id] = (current_idx, current_idx + count)
            current_idx += count
            
    print(f"✅ Total windows mapped: {current_idx}")
    
    # DEBUG: Show exactly how the PDB IDs are formatted from the folder
    mapped_ids_list = list(pdb_to_indices.keys())
    print(f"🔍 DEBUG: First 5 mapped PDB IDs from folder: {mapped_ids_list[:5]}")

    total_samples = current_idx
    
    for k in range(1, 11):
        def read_pdb_list(filepath):
            if not os.path.exists(filepath): 
                return set()
            with open(filepath, 'r') as f:
                return set(line.strip() for line in f if line.strip())
                
        # Exact file names as you specified
        train_file = os.path.join(FOLDS_DIR, f"train_fold{k}_pdb.txt")
        valid_file = os.path.join(FOLDS_DIR, f"valid_fold{k}_pdb.txt")
        test_file = os.path.join(FOLDS_DIR, f"test_fold{k}_pdb.txt")

        train_pdbs = read_pdb_list(train_file)
        valid_pdbs = read_pdb_list(valid_file)
        test_pdbs = read_pdb_list(test_file)
        
        # DEBUG: Show exactly how the PDB IDs are formatted in the text file
        if k == 1:
            train_list = list(train_pdbs)
            print(f"🔍 DEBUG: First 5 PDB IDs in train_fold1_pdb.txt: {train_list[:5]}")
            
        # Check overlap
        mapped_keys = set(pdb_to_indices.keys())
        train_overlap = len(train_pdbs.intersection(mapped_keys))
        valid_overlap = len(valid_pdbs.intersection(mapped_keys))
        test_overlap = len(test_pdbs.intersection(mapped_keys))
        
        print(f"🔍 Fold {k} Debug: Train overlap={train_overlap}, Valid overlap={valid_overlap}, Test overlap={test_overlap}")
        
        train_mask = np.zeros(total_samples, dtype=bool)
        valid_mask = np.zeros(total_samples, dtype=bool)
        test_mask = np.zeros(total_samples, dtype=bool)
        
        for pdb_id, (start, end) in pdb_to_indices.items():
            if pdb_id in train_pdbs: train_mask[start:end] = True
            if pdb_id in valid_pdbs: valid_mask[start:end] = True
            if pdb_id in test_pdbs: test_mask[start:end] = True
            
        np.save(os.path.join(OUTPUT_DIR, f"train_mask_fold{k}.npy"), train_mask)
        np.save(os.path.join(OUTPUT_DIR, f"valid_mask_fold{k}.npy"), valid_mask)
        np.save(os.path.join(OUTPUT_DIR, f"test_mask_fold{k}.npy"), test_mask)
        
        print(f"✅ Fold {k}: Train={train_mask.sum()}, Valid={valid_mask.sum()}, Test={test_mask.sum()}")
        
    print("-" * 70)
    print("🎉 Index mapping complete!")

if __name__ == "__main__":
    main()