import argparse
import numpy as np
import sys
from datasets import load_dataset, Dataset
from collections import Counter
from typing import Tuple, Optional, Dict

# for visualization
import math
import matplotlib.pyplot as plt


def get_patches_column_major(data_array: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """
    Pads a 2D array and extracts patches in column-major order.

    Given a 2D (n, t) array and a patch size (p0, p1), this function:
    1. Pads the array with zeros so that n is divisible by p0 and t is divisible by p1.
    2. Extracts all (p0, p1) patches.
    3. Returns the patches as a 3D array (num_patches, p0, p1) ordered in column-major fashion.

    Args:
        data_array: The 2D input NumPy array (e.g., dtype uint8).
        patch_size: A tuple (p0, p1) specifying the patch dimensions.

    Returns:
        A 3D NumPy array (num_patches, p0, p1) containing all
        extracted patches in column-major order.
    """
    if data_array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional, but got {data_array.ndim} dimensions.")
        
    p0, p1 = patch_size
    if not (p0 > 0 and p1 > 0):
        raise ValueError(f"Patch dimensions must be positive, but got ({p0}, {p1}).")

    n, t = data_array.shape

    # 1. Calculate padding
    pad_n = (p0 - (n % p0)) % p0
    pad_t = (p1 - (t % p1)) % p1

    # 2. Apply padding if needed
    if pad_n > 0 or pad_t > 0:
        padded_array = np.pad(
            data_array,
            ((0, pad_n), (0, pad_t)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_array = data_array
    
    # Get the new, padded dimensions
    N, T = padded_array.shape
    
    # Calculate the number of patches along each dimension
    num_patches_n = N // p0
    num_patches_t = T // p1

    # Reshape into a 4D array: (num_patches_n, p0, num_patches_t, p1)
    reshaped = padded_array.reshape(num_patches_n, p0, num_patches_t, p1)

    # Transpose to: (num_patches_t, num_patches_n, p0, p1)
    # This groups by patch-column, then patch-row.
    transposed = reshaped.transpose(2, 0, 1, 3)
    
    # Reshape to flatten patch indices (num_patches_t, num_patches_n) into a single dimension, giving (total_patches, p0, p1).
    # This preserves the column-major order.
    patches = transposed.reshape(-1, p0, p1)

    return patches

def process_and_count_patches(
    dataset_name: str,
    split: str,
    column_name: str,
    patch_size: Tuple[int, int]
) -> Optional[Counter]:
    """
    Iterates over a dataset, extracts patches, and counts their occurrences.

    Args:
        dataset_name: The name of the Hugging Face dataset.
        split: The dataset split to use (e.g., "train").
        column_name: The name of the column containing the 2D (n, t) arrays.
        patch_size: The (p0, p1) dimensions for patch extraction.

    Returns:
        A collections.Counter object where keys are the byte representations
        of unique patches and values are their total counts. Returns None on error.
    """
    print(f"Loading dataset '{dataset_name}' (split: {split}) with streaming...")
    # Use streaming=True to avoid downloading the entire dataset at once
    ds = load_dataset(dataset_name, split=split, streaming=True)

    patch_counts = Counter()
    total_rows = 0
    p0, p1 = patch_size
    
    print(f"Starting patch extraction for patch_size={patch_size} from column '{column_name}'...")

    for row in ds:
        total_rows += 1

        # Convert data to a numpy array
        data_array = np.array(row[column_name], dtype=np.uint8)
        
        # Get ALL patches for this row
        all_patches_in_row = get_patches_column_major(data_array, patch_size)
        
        # Add them to the counter
        # We must convert arrays to a hashable type (bytes) to use as dict keys
        for patch in all_patches_in_row:
            patch_bytes = patch.tobytes()
            patch_counts[patch_bytes] += 1
        
        if total_rows % 1 == 0:
            print(f"Processed {total_rows} rows. Found {len(patch_counts)} unique patches so far.")

    print("\n...Processing complete.")
    print(f"Total rows processed: {total_rows}")
    print(f"Total unique patches (p0={p0}, p1={p1}) found: {len(patch_counts)}")
    
    return patch_counts

def reconstruct_patch(patch_bytes: bytes, patch_size: Tuple[int, int], dtype=np.uint8) -> np.ndarray:
    """Helper function to convert patch bytes back into a NumPy array for inspection."""
    # Note: Adjust dtype if your original data array is not uint8
    try:
        return np.frombuffer(patch_bytes, dtype=dtype).reshape(patch_size)
    except ValueError:
        return np.array([[f"Error: could not reshape bytes to {patch_size} with dtype {dtype}."]])

def visualize_top_k_patches(patch_counts: Counter, patch_size: Tuple[int, int], dtype: np.dtype, k: int):
    """
    Visualizes the top K most common patches in a grid.

    Args:
        patch_counts: Counter object with patch_bytes as keys and counts as values.
        patch_size: The (p0, p1) dimensions of each patch.
        dtype: The NumPy dtype of the original patch data.
        k: The number of top patches to display.
    """    
    top_patches = patch_counts.most_common(k)
    if not top_patches:
        print("No patches were found to visualize.")
        return

    # Determine grid size (e.g., 4x4 for k=16)
    nrows = int(math.ceil(math.sqrt(k)))
    ncols = nrows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    fig.suptitle(f"Top {k} most common patches ({patch_size[0]}x{patch_size[1]})", fontsize=16)

    # Flatten axes array for easy iteration, regardless of nrows/ncols
    axes_flat = axes.flatten()

    for i, (patch_bytes, count) in enumerate(top_patches):
        patch_array = reconstruct_patch(patch_bytes, patch_size, dtype)
        
        ax = axes_flat[i]
        
        # Display the patch
        ax.imshow(patch_array, cmap='jet', vmin=0, vmax=1)
        
        ax.set_title(f"#{i+1}\n(Count: {count})", fontsize=10)
        ax.set_axis_off() # Hide x and y axes

    # Hide any unused subplots
    for i in range(len(top_patches), len(axes_flat)):
        axes_flat[i].set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(f"rodent_({patch_size[0]}x{patch_size[1]})_motifs.jpeg", bbox_inches='tight')

def save_patches_to_dataset(
    patch_counts: Counter,
    patch_size: Tuple[int, int],
    dtype: np.dtype,
    repo_id: str
):
    """
    Saves the unique patches and their frequencies to a Hugging Face Hub dataset.

    Args:
        patch_counts: Counter object from processing.
        patch_size: The (p0, p1) dimensions of each patch.
        dtype: The NumPy dtype of the original patch data.
        repo_id: The Hugging Face Hub repo-id (e.g., "username/dataset-name").
    """
    print(f"\n--- Saving Unique Patches to Hugging Face Hub ---")

    total_patches = sum(patch_counts.values())

    print(f"Total unique patches: {len(patch_counts)}")
    print(f"Total patches (all occurrences): {total_patches}")

    # Prepare data for the dataset
    data_list = []
    for patch_bytes, count in patch_counts.items():
        patch_array = reconstruct_patch(patch_bytes, patch_size, dtype)
        probability = count / total_patches
        
        data_list.append({
            "patch": patch_array.flatten().tolist(), # Convert NumPy array to FLAT list
            "frequency": count,
            "probability": probability
        })

    # Sort by frequency (most common first)
    data_list.sort(key=lambda x: x['frequency'], reverse=True)

    print("Creating dataset object...")
    # Create the dataset
    unique_patch_dataset = Dataset.from_list(data_list)
    
    # Push to Hub
    print(f"Pushing dataset to Hugging Face Hub at '{repo_id}'...")
    unique_patch_dataset.push_to_hub(repo_id, token=True)
    
    print("...Push complete.")
    print(f"You can now access the dataset at:")
    print(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract motifs from the rodent pile.")
    parser.add_argument("--hf_repo_id", type=str, default="eminorhan/neural-pile-rodent-reordered-1x15", help="The Hugging Face Repo ID for remote saving.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[1, 15], metavar=('HEIGHT', 'WIDTH'), help="The patch size as two integers (e.g., --patch-size 1 15).")
    args = parser.parse_args()

    # From argparse
    HF_REPO_ID = args.hf_repo_id  # repo id for remote saving
    PATCH_SIZE = tuple(args.patch_size) # convert list [p0, p1] to tuple (p0, p1)

    # Other arguments
    DATASET_NAME = "eminorhan/neural-pile-rodent-reordered"
    DATASET_SPLITS = ["train", "test"]  # including both splits    
    DATA_COLUMN = "spike_counts"        
    DATA_DTYPE = np.uint8 
    K_TOP_PATCHES = 16  # top-k visualization: e.g., 16 for a 4x4 grid

    counts = Counter()

    # Iterate over both splits and accumulate counts
    for split in DATASET_SPLITS:
        print(f"\n--- Processing Split: {split} ---")
        split_counts = process_and_count_patches(
            dataset_name=DATASET_NAME,
            split=split,
            column_name=DATA_COLUMN,
            patch_size=PATCH_SIZE
        )
        counts.update(split_counts)
    
    print("\n--- Top 10 Most Common Motifs (Aggregated) ---")
    for i, (patch_bytes, count) in enumerate(counts.most_common(10)):
        print(f"\n#{i+1}: Count = {count}")
        
        # Uncomment the lines below if you want to print the actual patch array
        # patch_array = reconstruct_patch(patch_bytes, PATCH_SIZE, dtype=DATA_DTYPE)
        # print(patch_array)
        
        # For brevity, we'll just print a snippet of the byte representation
        print(f"   (Patch bytes snippet: {patch_bytes.hex()[:40]}...)")

    # Visualize top-K patches
    visualize_top_k_patches(
        patch_counts=counts,
        patch_size=PATCH_SIZE,
        dtype=DATA_DTYPE,
        k=K_TOP_PATCHES
    )

    # Save dataset to hub
    save_patches_to_dataset(
        patch_counts=counts,
        patch_size=PATCH_SIZE,
        dtype=DATA_DTYPE,
        repo_id=HF_REPO_ID
    )