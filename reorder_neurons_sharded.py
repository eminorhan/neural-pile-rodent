import os
import glob
import shutil
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from datasets import load_dataset, Dataset

# --- 1. Optimization Settings ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- 2. Core Logic ---
def reorder_neurons(spike_counts, method='cosine'):
    spike_counts = np.array(spike_counts)
    if spike_counts.shape[0] < 2: return spike_counts
    try:
        dists = pdist(spike_counts, metric=method)
        dists = np.nan_to_num(dists, nan=2.0)
        linkage_matrix = linkage(dists, method='ward', optimal_ordering=True)
        sorted_indices = leaves_list(linkage_matrix)
        return spike_counts[sorted_indices, :]
    except Exception:
        return spike_counts

# --- 3. Worker Logic (Must be Top-Level) ---
def process_batch_of_shards(batch_of_shards, output_root):
    """
    Worker function to process a list of arrow files.
    """
    print(f"Worker {os.getpid()} processing batch of {len(batch_of_shards)} shards...")
    
    for shard_info in batch_of_shards:
        # shard_info tuple: (split_name, arrow_file_path)
        split, arrow_path = shard_info
        
        try:
            # Create output directory
            save_dir = os.path.join(output_root, split)
            # makedirs is thread-safe enough for this, but could race rarely. 
            # exist_ok=True handles the race condition.
            os.makedirs(save_dir, exist_ok=True)
            
            # Create a deterministic filename based on the arrow filename
            base_name = os.path.splitext(os.path.basename(arrow_path))[0]
            output_path = os.path.join(save_dir, f"{base_name}.parquet")

            # 1. Read Arrow file directly
            ds_shard = Dataset.from_file(arrow_path)
            
            # 2. Convert to Pandas
            df = ds_shard.to_pandas()

            # 3. Apply Reordering
            df['spike_counts'] = df['spike_counts'].apply(reorder_neurons)

            # 4. Save to Parquet
            df.to_parquet(output_path, index=False)
            
        except Exception as e:
            print(f"Failed to process {split} shard {arrow_path}: {e}")

def main():
    SOURCE_REPO = "eminorhan/neural-pile-rodent"
    TARGET_REPO = "eminorhan/neural-pile-rodent-reordered"
    NUM_PROC = 120
    TEMP_ROOT = "processed_shards_temp"

    if os.path.exists(TEMP_ROOT): shutil.rmtree(TEMP_ROOT)
    os.makedirs(TEMP_ROOT)

    print(f"Loading dataset metadata from {SOURCE_REPO}...")
    ds = load_dataset(SOURCE_REPO)

    # --- Collect Physical Arrow Files ---
    tasks = []
    
    if 'train' in ds:
        # Access internal file list from the cached dataset
        train_files = [x['filename'] for x in ds['train'].cache_files]
        print(f"Found {len(train_files)} backing Arrow files for 'train'.")
        for f in train_files:
            tasks.append(('train', f))

    if 'test' in ds:
        test_files = [x['filename'] for x in ds['test'].cache_files]
        print(f"Found {len(test_files)} backing Arrow files for 'test'.")
        for f in test_files:
            tasks.append(('test', f))
            
    total_shards = len(tasks)
    print(f"Total physical shards to process: {total_shards}")

    if total_shards == 0:
        print("Error: No cache files found.")
        return

    # --- Distribute Work ---
    # Create batches of tasks for the workers
    batches_for_procs = np.array_split(tasks, NUM_PROC)
    # Convert numpy arrays back to python lists
    batches_for_procs = [b.tolist() for b in batches_for_procs if len(b) > 0]
    
    print(f"Starting {len(batches_for_procs)} processes on {NUM_PROC} cores...")

    # Bind the output_root argument so the function is ready for map
    worker_func = partial(process_batch_of_shards, output_root=TEMP_ROOT)

    # --- Parallel Execution ---
    with multiprocessing.Pool(NUM_PROC) as pool:
        pool.map(worker_func, batches_for_procs)

    # --- Reassemble and Push ---
    print("Reassembling dataset from processed Parquet files...")
    data_files = {}
    
    train_shards = sorted(glob.glob(f"{TEMP_ROOT}/train/*.parquet"))
    if train_shards: data_files["train"] = train_shards
    
    test_shards = sorted(glob.glob(f"{TEMP_ROOT}/test/*.parquet"))
    if test_shards: data_files["test"] = test_shards

    # Reload using the parquet files we just wrote
    final_ds = load_dataset("parquet", data_files=data_files)
    
    print(f"Pushing to {TARGET_REPO}...")
    final_ds.push_to_hub(TARGET_REPO, token=True, max_shard_size="1GB")
    
    # Cleanup
    shutil.rmtree(TEMP_ROOT)
    print("Done.")

if __name__ == "__main__":
    main()