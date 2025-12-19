import numpy as np
from datasets import load_dataset
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm
import os

def process_dataset_shard(args):
    """
    Worker function to process a specific shard of the dataset.
    This runs in a separate process.
    """
    rank, num_shards, dataset_name, split, column, batch_size = args
    
    # Initialize local histogram
    local_histogram = np.zeros(256, dtype=np.int64)
    
    try:
        # Each worker needs to instantiate its own connection to the dataset
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        # Shard the dataset
        try:
            sharded_dataset = dataset.shard(num_shards=num_shards, index=rank)
        except (IndexError, ValueError):
            # IndexError: Happens when num_workers > number of underlying files/shards.
            # ValueError: Happens if the dataset is not shardable in the requested way.
            # In either case, this worker has no data to process.
            return local_histogram
        
        # Iterate through this shard
        # Use tqdm to show progress statistics (speed, count) for this worker
        # mininterval ensures we don't flood the console with updates from many workers
        iterator = tqdm(
            sharded_dataset.iter(batch_size=batch_size),
            desc=f"Worker {rank}",
            unit="batch",
            mininterval=2.0
        )
        
        for batch in iterator:
            try:
                # Check if column exists in this batch
                if column not in batch:
                    continue

                # Extract data
                raw_data = batch[column]
                
                # If raw_data is empty or None
                if not raw_data:
                    continue

                # Extract and flatten data
                # np.asanyarray ensures we handle lists or existing arrays efficiently
                batch_data = [np.asanyarray(x).ravel() for x in raw_data]
                
                if not batch_data:
                    continue
                
                flat_batch = np.concatenate(batch_data)
                
                # Vectorized count
                counts = np.bincount(flat_batch, minlength=256)
                
                # Accumulate
                if len(counts) > 256:
                    local_histogram += counts[:256]
                else:
                    local_histogram += counts
            except Exception:
                # Skip bad batches without crashing the worker
                continue
                
    except Exception as e:
        print(f"Worker {rank} failed with unexpected error: {e}")
        return np.zeros(256, dtype=np.int64)
        
    return local_histogram

def estimate_spike_probabilities(dataset_name, split="train", column="spike_counts", batch_size=1000, num_workers=None):
    """
    Parallelized estimation of spike count probabilities.
    """
    if num_workers is None:
        num_workers = cpu_count() // 2
        
    print(f"Starting parallel processing with {num_workers} workers...")
    print(f"Dataset: {dataset_name} | Batch Size per worker: {batch_size}")

    # Prepare arguments for each worker
    # Format: (rank, total_shards, dataset_name, split, column, batch_size)
    worker_args = [(i, num_workers, dataset_name, split, column, batch_size) for i in range(num_workers)]

    # Use 'spawn' context instead of default (which might be 'fork')
    ctx = get_context('spawn')

    # Spawn processes
    with ctx.Pool(num_workers) as pool:
        # Map the worker function to the arguments
        results = pool.map(process_dataset_shard, worker_args)
    
    print("All workers finished. Aggregating results...")

    # Sum up histograms from all workers
    global_histogram = np.sum(results, axis=0)

    # Calculate probabilities
    total_spikes = np.sum(global_histogram)
    
    if total_spikes == 0:
        print("No spike data found.")
        return np.zeros(256)
        
    probabilities = global_histogram / total_spikes
    
    return probabilities

if __name__ == "__main__":

    DATASET_NAME = "eminorhan/neural-pile-rodent" 
    COLUMN_NAME = "spike_counts"
    BATCH_SIZE = 8 
    
    # Run parallel estimation
    probs = estimate_spike_probabilities(DATASET_NAME, column=COLUMN_NAME, batch_size=BATCH_SIZE)

    # Save token probabilities 
    np.save("rodent_token_probabilities.npy", probs)

    if probs is not None:
        print("\n--- Results ---")
        print(f"Probability Vector Shape: {probs.shape}")
        print(f"Sum of probabilities: {np.sum(probs):.6f}") 
        
        print("\nFirst 10 probabilities (0-9):")
        print(probs[:10])
        
        print(f"\nProbability of 0 spikes: {probs[0]:.6f}")
        print(f"Probability of 255 spikes: {probs[255]:.6f}")