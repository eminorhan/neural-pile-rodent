import argparse
import numpy as np
import pickle
from datasets import load_dataset
from typing import Dict, Tuple
import os
import re
from tqdm import tqdm

def compute_similarity_batch(query_vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    """
    Computes a similarity metric between a batch of query vectors and reference vectors.
    
    Args:
        query_vectors: Shape (batch_size, features)
        reference_vectors: Shape (num_references, features)
    
    Returns:
        indices: Shape (batch_size,) - The index of the reference vector with highest dot product.
    """
    # Compute Dot Product directly (Unnormalized)
    # (Batch, Feat) @ (Feat, Refs) -> (Batch, Refs)
    similarity_matrix = np.dot(query_vectors, reference_vectors.T)
    
    # Find Argmax
    # Returns the index (0 to K) of the maximum dot product for each query
    best_indices = np.argmax(similarity_matrix, axis=1)
    
    return best_indices

def assign_indices_and_save(
    repo_id: str,
    k_limit: int,
    output_path: str,
    dtype: np.dtype = np.uint8,
    batch_size: int = 1000
):
    """
    Assigns indices 0-K to dataset rows based on rank or cosine similarity.
    
    1. Loads the dataset from Hugging Face.
    2. Assigns indices 0 to K to the first K+1 rows.
    3. For remaining rows, finds the closest match in the first K+1 rows via cosine sim.
    4. Saves the mapping {patch_bytes: integer_index} to disk via Pickle.
    """
    
    print(f"Loading dataset from Hugging Face: {repo_id}...")
    try:
        ds = load_dataset(repo_id, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_rows = len(ds)
    match = re.search(r"(\d+)x(\d+)$", repo_id)
    patch_size = (int(match.group(1)), int(match.group(2)))
    print(f"Dataset loaded. Total unique patches: {total_rows}; patch shape: {patch_size}")
    
    # Ensure K is not larger than the dataset
    actual_k_limit = min(k_limit, total_rows - 1)
    
    # --- STEP 1: PREPARE REFERENCE SET (0 to K) ---
    print(f"Extracting reference patches (indices 0 to {actual_k_limit})...")
    
    # Slice the dataset to get the first K+1 rows
    ref_data = ds.select(range(actual_k_limit + 1))
    
    # Convert lists to numpy array (Shape: K+1, Flattened_Dim)
    ref_arrays = np.array(ref_data['patch'], dtype=np.float32)
    print(f"Ref_arrays shape: {ref_arrays.shape}")

    # Pre-normalize reference vectors for fast cosine similarity later
    ref_norms = np.linalg.norm(ref_arrays, axis=1, keepdims=True)
    ref_norms[ref_norms == 0] = 1e-10
    ref_vectors_normalized = ref_arrays / ref_norms
    
    # Initialize the result dictionary
    patch_index_map: Dict[bytes, int] = {}
    
    # Add the reference patches to the map
    print("Assigning indices to reference patches...")
    for i in range(actual_k_limit + 1):
        # We must cast back to the original dtype (e.g., uint8) before bytes conversion
        # to ensure the key matches the original byte representation.
        original_patch = ref_arrays[i].astype(dtype)
        patch_bytes = original_patch.tobytes()
        patch_index_map[patch_bytes] = i
        
    # --- STEP 2: PROCESS REMAINING ROWS (K+1 to End) ---
    start_idx = actual_k_limit + 1
    
    if start_idx < total_rows:
        print(f"Processing remaining patches ({start_idx} to {total_rows})...")
        
        # We iterate in batches for memory efficiency
        for i in tqdm(range(start_idx, total_rows, batch_size)):
            end_idx = min(i + batch_size, total_rows)
            
            # Load batch
            batch_data = ds.select(range(i, end_idx))
            batch_arrays = np.array(batch_data['patch'], dtype=np.float32)
            
            # Find closest reference indices
            assigned_indices = compute_similarity_batch(batch_arrays, ref_vectors_normalized)
            
            # Store in dictionary
            for j, index_assignment in enumerate(assigned_indices):
                # Reconstruct bytes key
                original_patch = batch_arrays[j].astype(dtype)
                patch_bytes = original_patch.tobytes()
                
                # Store the assignment (int)
                patch_index_map[patch_bytes] = int(index_assignment)
    else:
        print("No remaining rows to process (K >= total_rows).")

    # --- STEP 3: SAVE TO DISK ---
    print(f"\nSaving dictionary to '{output_path}' using Pickle...")
    # Prepare the data structure to save
    output_data = {
        "index_map": patch_index_map,
        "patch_size": patch_size,
        "k_limit": actual_k_limit
    }
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    print("Save complete.")
    print(f"Dictionary size: {len(patch_index_map)} entries.")
    print(f"Metadata saved: Patch Size={patch_size}, K Limit={actual_k_limit}")

def load_index_dictionary(path: str) -> Dict[bytes, int]:
    """Helper function to demonstrate how to load the saved dictionary."""
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Embarrasingly simple tokenizer for neural pile motifs.")
    parser.add_argument("--hf_repo_id", type=str, default="eminorhan/neural-pile-rodent-1x15", help="Hugging Face repo ID for loading the motifs.")
    parser.add_argument("--output_filename", type=str, default="tokenizer_rodent_1x15_32k.pkl", help="Output file name where the motif index map will be saved.")
    parser.add_argument("--k_limit", type=int, default=32_000-2, help="Max index to be used for encoding the motifs.")
    args = parser.parse_args()

    # From argparse
    HF_REPO_ID = args.hf_repo_id
    OUTPUT_FILENAME = args.output_filename
    K_LIMIT = args.k_limit

    # Other arguments
    PATCH_DTYPE = np.uint8 

    print("--- Patch Index Assigner ---")
    print(f"Repo: {HF_REPO_ID}")
    print(f"K Limit: {K_LIMIT}")
    print("---")
    
    assign_indices_and_save(
        repo_id=HF_REPO_ID,
        k_limit=K_LIMIT,
        output_path=OUTPUT_FILENAME,
        dtype=PATCH_DTYPE
    )
    
    # --- VERIFICATION ---
    # Uncomment to test loading
    print("\nVerifying load...")
    loaded_map = load_index_dictionary(OUTPUT_FILENAME)
    print(f"Successfully loaded map with {len(loaded_map)} keys.")