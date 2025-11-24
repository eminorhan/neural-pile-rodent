import numpy as np
import pickle
from datasets import load_dataset
from typing import Dict, Tuple
import os
from tqdm import tqdm

def compute_cosine_similarity_batch(query_vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a batch of query vectors and reference vectors.
    
    Args:
        query_vectors: Shape (batch_size, features)
        reference_vectors: Shape (num_references, features) - Must be pre-normalized!
    
    Returns:
        indices: Shape (batch_size,) - The index of the most similar reference vector for each query.
    """
    # 1. Normalize Query Vectors
    # Compute norms (add epsilon to avoid division by zero)
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_norms[query_norms == 0] = 1e-10
    query_normalized = query_vectors / query_norms
    
    # 2. Compute Dot Product (Cosine Similarity)
    # (Batch, Feat) @ (Feat, Refs) -> (Batch, Refs)
    # Since inputs are normalized, dot product == cosine similarity
    similarity_matrix = np.dot(query_normalized, reference_vectors.T)
    
    # 3. Find Argmax
    # Returns the index (0 to K) of the maximum similarity for each query
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
    print(f"Dataset loaded. Total unique patches: {total_rows}")
    
    # Ensure K is not larger than the dataset
    actual_k_limit = min(k_limit, total_rows - 1)
    
    # --- STEP 1: PREPARE REFERENCE SET (0 to K) ---
    print(f"Extracting reference patches (indices 0 to {actual_k_limit})...")
    
    # Slice the dataset to get the first K+1 rows
    ref_data = ds.select(range(actual_k_limit + 1))
    
    # Convert lists to numpy array (Shape: K+1, Flattened_Dim)
    ref_arrays = np.array(ref_data['patch'], dtype=np.float32)
    
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
            assigned_indices = compute_cosine_similarity_batch(batch_arrays, ref_vectors_normalized)
            
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
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(patch_index_map, f)
        print("Save complete.")
        print(f"Dictionary size: {len(patch_index_map)} entries.")
    except Exception as e:
        print(f"Error saving file: {e}")

def load_index_dictionary(path: str) -> Dict[bytes, int]:
    """Helper function to demonstrate how to load the saved dictionary."""
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    
    HF_REPO_ID = "eminorhan/neural-pile-rodent-1x10"
    K_LIMIT = 128_000 - 2 
    OUTPUT_FILENAME = "patch_index_map.pkl"
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
    # print("\nVerifying load...")
    # loaded_map = load_index_dictionary(OUTPUT_FILENAME)
    # print(f"Successfully loaded map with {len(loaded_map)} keys.")