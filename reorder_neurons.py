import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from datasets import load_dataset, DatasetDict

def reorder_neurons(spike_counts, method='cosine'):
    """
    Reorders neurons based on hierarchical clustering.
    Returns the sorted spike count array.
    """
    # Ensure input is a numpy array
    spike_counts = np.array(spike_counts)
    
    # Safety check for small arrays
    if spike_counts.shape[0] < 2:
        return spike_counts

    try:
        # Calculate distance
        dists = pdist(spike_counts, metric=method)
        # Handle NaN distances (common with silent neurons)
        # 2.0 is the max distance for correlation (range -1 to 1)
        dists = np.nan_to_num(dists, nan=2.0)
        
        # Cluster
        linkage_matrix = linkage(dists, method='ward', optimal_ordering=True)
        sorted_indices = leaves_list(linkage_matrix)
        
        return spike_counts[sorted_indices, :]
    except Exception as e:
        # Fallback if clustering fails
        return spike_counts

# 2. Define the processing wrapper for the map function
def apply_reordering(example):
    """
    Wrapper to apply reordering to a specific row.
    """
    # Convert list/tensor to numpy, reorder, and convert back to list 
    # (HuggingFace datasets store arrays as lists internally)
    original_spikes = example['spike_counts']
    reordered_spikes = reorder_neurons(original_spikes)
    
    # Update the example
    example['spike_counts'] = reordered_spikes
    return example

def main():

    os.environ["OMP_NUM_THREADS"] = "1"

    # Configuration
    SOURCE_REPO = "eminorhan/neural-pile-rodent"
    TARGET_REPO = "eminorhan/neural-pile-rodent-reordered"
    NUM_PROC = 72  # Adjust based on available CPU cores
    
    print(f"Loading dataset from {SOURCE_REPO}...")
    dataset = load_dataset(SOURCE_REPO)
    
    print("Dataset loaded. Structure:")
    print(dataset)

    # 3. Apply the function to all splits (train/test) in parallel
    print(f"Reordering neurons using {NUM_PROC} processes...")
    reordered_dataset = dataset.map(
        apply_reordering,
        num_proc=NUM_PROC,
        desc="Reordering neurons"
    )

    # 4. Push to Hugging Face Hub
    print(f"Pushing processed dataset to {TARGET_REPO}...")
    reordered_dataset.push_to_hub(TARGET_REPO, max_shard_size="1GB", token=True)
    
    print("Done! Dataset successfully uploaded.")

if __name__ == "__main__":
    main()