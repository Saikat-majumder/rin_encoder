import os
import glob
import numpy as np
import gc

CACHE_DIR = "pipeline_cache"
OUTPUT_FILE = "closeness_fingerprints.npy"

def combine_cache():
    chunk_files = sorted(glob.glob(os.path.join(CACHE_DIR, "chunk_*.npy")))
    if not chunk_files:
        print("❌ No cached chunks found. Run pdb_parsing.py first.")
        return

    print(f"Found {len(chunk_files)} cached chunks. Combining...")
    
    # Load first chunk to get the shape (16384 for 128x128)
    first_chunk = np.load(chunk_files[0])
    feature_dim = first_chunk.shape[1]
    print(f"   Feature dimension per window: {feature_dim}")
    
    # Calculate total size to pre-allocate memory efficiently
    total_samples = sum(np.load(f, mmap_mode='r').shape[0] for f in chunk_files)
    print(f"   Total samples to combine: {total_samples}")
    
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
    print(f"   Final shape: {final_X.shape}")
    
    # Clean up cache to save disk space
    print("   Cleaning up cache directory...")
    for f in chunk_files:
        os.remove(f)
    # Remove .done flags
    for f in glob.glob(os.path.join(CACHE_DIR, "*.done")):
        os.remove(f)
    os.rmdir(CACHE_DIR)
    print("   Cache cleared. Ready for training!")

if __name__ == "__main__":
    combine_cache()