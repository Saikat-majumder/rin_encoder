#!/usr/bin/env python3
"""
run_pipeline.py

One-click pipeline to batch process PDB files into sliding windows of 
Probabilistic Distance Matrices (PDM), cache them safely to disk to 
prevent RAM crashes, and combine them into the final training dataset.

Parameters: Window Size = 128, Stride = 64
"""

import os
import glob
import numpy as np
import gc
from rin_encodermap.pdb_parsing import build_pdm_fingerprint_dataset

# --- CONFIGURATION ---
PDB_DIR = "/home/subhadip/rin_encoder_1/PDB-Uniprot"
CACHE_DIR = "pipeline_cache"
OUTPUT_FILE = "closeness_fingerprints.npy"
WINDOW_SIZE = 128  # Must match pdb_parsing.py

def combine_cache():
    """Reads cached chunks from disk and combines them into the final .npy file."""
    chunk_files = sorted(glob.glob(os.path.join(CACHE_DIR, "chunk_*.npy")))
    if not chunk_files:
        print("❌ No cached chunks found to combine.")
        return False

    print(f"\n🔄 Found {len(chunk_files)} cached chunks. Combining into final dataset...")
    
    # Load first chunk to get the shape (16384 for 128x128)
    first_chunk = np.load(chunk_files[0])
    feature_dim = first_chunk.shape[1]
    print(f"   • Feature dimension per window: {feature_dim} ({WINDOW_SIZE}x{WINDOW_SIZE})")
    
    # Calculate total size to pre-allocate memory efficiently
    total_samples = sum(np.load(f, mmap_mode='r').shape[0] for f in chunk_files)
    print(f"   • Total samples to combine: {total_samples}")
    
    final_X = np.empty((total_samples, feature_dim), dtype=np.float32)
    
    current_idx = 0
    for f in chunk_files:
        chunk = np.load(f)
        n = chunk.shape[0]
        final_X[current_idx:current_idx+n] = chunk
        current_idx += n
        gc.collect() # Clear memory after each load

    # Save the final combined dataset
    np.save(OUTPUT_FILE, final_X)
    print(f"\n🎉 Successfully combined into '{OUTPUT_FILE}'")
    print(f"   • Final shape: {final_X.shape}")
    
    # Clean up cache to save disk space
    print("   • Cleaning up cache directory...")
    for f in chunk_files:
        os.remove(f)
    # Remove .done flags
    for f in glob.glob(os.path.join(CACHE_DIR, "*.done")):
        os.remove(f)
    
    # Remove directory if empty
    if not os.listdir(CACHE_DIR):
        os.rmdir(CACHE_DIR)
        
    print("   • Cache cleared. Ready for training!")
    return True

def main():
    pdb_pattern = os.path.join(PDB_DIR, "*.pdb")
    
    # Check if directory exists
    if not os.path.isdir(PDB_DIR):
        print(f"❌ Error: Directory '{PDB_DIR}' not found.")
        return

    # Get list of files
    pdb_files = glob.glob(pdb_pattern)
    if not pdb_files:
        print(f"❌ Error: No .pdb files found in '{PDB_DIR}'.")
        return

    print("="*60)
    print("🚀 STARTING PPI DISTANCE MATRIX PIPELINE")
    print(f"   • Source: {PDB_DIR}")
    print(f"   • Window Size: {WINDOW_SIZE} | Stride: 64")
    print("="*60)

    try:
        # STEP 1: Parse PDBs and generate chunks (Memory Safe)
        print("\n[STEP 1/2] Processing PDB files and generating sliding windows...")
        build_pdm_fingerprint_dataset(pdb_pattern)
        
        # STEP 2: Combine chunks into final dataset
        print("\n[STEP 2/2] Combining cached chunks...")
        success = combine_cache()
        
        if success:
            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETE!")
            print("="*60)
            print("You can now train the original EncoderMap model using:")
            print(f"python rin_encodermap/train_encodermap.py \\")
            print(f"    --input {OUTPUT_FILE} \\")
            print(f"    --output_dir runs/ppi_baseline_128 \\")
            print(f"    --input_dim {WINDOW_SIZE * WINDOW_SIZE} \\")
            print(f"    --bottleneck_dim 32 \\")
            print(f"    --n_steps 5000")
            
    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
        print("💡 Tip: If your system crashed, just run this script again. It will resume from the cache.")

if __name__ == "__main__":
    main()