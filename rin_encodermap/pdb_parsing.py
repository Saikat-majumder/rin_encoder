import os
import glob
import numpy as np
import gc
from scipy.spatial.distance import cdist

# --- UPDATED PARAMETERS ---
WINDOW_SIZE = 128
STRIDE = 64        # 50% overlap. Drastically reduces redundant windows.
CHUNK_SIZE = 5000    # Saves to disk every 5,000 windows (~320 MB per chunk). Keeps RAM safe.
CACHE_DIR = "pipeline_cache"

def parse_pdb_chains(filepath):
    """Extracts C-alpha coordinates and Chain IDs from a PDB file."""
    chains = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                if line[12:16].strip() == "CA":
                    chain_id = line[21]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if chain_id not in chains:
                        chains[chain_id] = []
                    chains[chain_id].append([x, y, z])
    
    sorted_keys = sorted(chains.keys())
    if len(sorted_keys) < 2:
        return None, None
        
    return np.array(chains[sorted_keys[0]]), np.array(chains[sorted_keys[1]])

def build_pdm_fingerprint_dataset(pdb_pattern):
    """Processes PDBs into sliding windows, caching to disk to prevent RAM crashes."""
    W = WINDOW_SIZE
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    pdb_files = sorted(glob.glob(pdb_pattern))
    print(f"Found {len(pdb_files)} PDB files.")
    
    chunk_idx = 0
    total_windows = 0
    X_chunk = []

    for idx, pdb_file in enumerate(pdb_files):
        base_name = os.path.basename(pdb_file)
        flag_file = os.path.join(CACHE_DIR, f"{base_name}.done")
        
        # --- CACHE CHECKPOINT: Skip if already processed ---
        if os.path.exists(flag_file):
            print(f"[{idx+1}/{len(pdb_files)}] ⏩ Skipping (already in cache): {base_name}")
            continue

        print(f"[{idx+1}/{len(pdb_files)}] Processing {base_name}...")
        
        coordsA, coordsB = parse_pdb_chains(pdb_file)
        if coordsA is None:
            print("   ⚠️ Skipping: Less than 2 chains found.")
            continue
            
        if len(coordsA) < W or len(coordsB) < W:
            print(f"   ⚠️ Skipping: Chains are smaller than window size {W}.")
            continue
            
        # 1. Calculate physical distance matrix
        pdm = cdist(coordsA, coordsB)
        
        # 2. Apply normalization: <=5A is 0, >=80A is 1, linear in between
        pdm_normalized = np.clip((pdm - 5.0) / 75.0, 0.0, 1.0)
        
        # 3. Apply Sliding Window
        windows_generated = 0
        for i in range(0, len(coordsA) - W + 1, STRIDE):
            for j in range(0, len(coordsB) - W + 1, STRIDE):
                pdm_window = pdm_normalized[i : i+W, j : j+W]
                X_chunk.append(pdm_window.flatten())
                windows_generated += 1
                total_windows += 1
                
                # 4. DISK CACHE: Save and clear memory when chunk is full
                if len(X_chunk) >= CHUNK_SIZE:
                    chunk_filename = os.path.join(CACHE_DIR, f"chunk_{chunk_idx:05d}.npy")
                    np.save(chunk_filename, np.array(X_chunk, dtype=np.float32))
                    print(f"   💾 Cached {chunk_filename} ({len(X_chunk)} windows). Clearing RAM...")
                    X_chunk = []  # Clear the list
                    gc.collect()  # Force Python garbage collection
                    chunk_idx += 1
                    
        print(f"   ✅ Generated {windows_generated} windows (W={W}, Stride={STRIDE}).")
        
        # Mark file as successfully processed in the cache
        with open(flag_file, 'w') as f:
            f.write("done")

    # Save any remaining windows in the final chunk
    if len(X_chunk) > 0:
        chunk_filename = os.path.join(CACHE_DIR, f"chunk_{chunk_idx:05d}.npy")
        np.save(chunk_filename, np.array(X_chunk, dtype=np.float32))
        print(f"   💾 Cached final {chunk_filename} ({len(X_chunk)} windows).")
        X_chunk = []
        gc.collect()
        chunk_idx += 1

    print(f"\n🎉 Dataset generation complete!")
    print(f"   Total windows generated: ~{total_windows}")
    print(f"   Cached in '{CACHE_DIR}/' directory.")
    print("   Run 'python combine_cache.py' to merge them for training.")

if __name__ == "__main__":
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else "/home/subhadip/rin_encoder_1/filtered_pdb/*.pdb"
    build_pdm_fingerprint_dataset(pattern)