import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_probabilities(npy_filepath, output_image):
    """
    Loads spike probabilities from an .npy file and plots them on a semilogy scale.
    
    Args:
        npy_filepath (str): Path to the numpy file containing the probability vector.
        output_image (str): Path where the resulting plot image will be saved.
    """
    if not os.path.exists(npy_filepath):
        print(f"Error: File '{npy_filepath}' not found.")
        print("Please ensure you have run the estimator script and uncommented the 'np.save' line.")
        return

    try:
        # Load the probability vector
        probs = np.load(npy_filepath)
        print(f"Loaded probabilities from {npy_filepath}. Shape: {probs.shape}")
    except Exception as e:
        print(f"Error reading .npy file: {e}")
        return

    # --- Weight Calculation ---
    weights = np.zeros_like(probs, dtype=np.float64)
    
    # Define the cutoff point
    cutoff = 10
    
    # 1. Calculate 1/sqrt(p) for spike counts 0 to cutoff
    # We add a tiny epsilon to avoid division by zero if a probability is exactly 0
    epsilon = 1e-9
    # Slicing is exclusive at the end, so cutoff+1 covers index cutoff
    weights[:cutoff+1] = 1.0 / (probs[:cutoff+1] + epsilon)
    
    # 2. For remaining spike counts (cutoff+1 to 255), clip them to the weight of count cutoff
    weights[cutoff+1:] = weights[cutoff]
    
    # Save the result to the requested file as a torch tensor
    weights_tensor = torch.from_numpy(weights).float() # Ensures float32
    output_pt_file = "rodent_token_weights_inv.pt"
    torch.save(weights_tensor, output_pt_file)
    print(f"Weights converted to Torch Tensor (float32) and saved to '{output_pt_file}'")
    print(f"Token weights: {weights_tensor}")
    # --------------------------

    # Create the x-axis (0 to 255)
    x = np.arange(len(probs))

    # Setup the plot
    plt.figure(figsize=(10, 6))
    
    # Plot using semilogy (logarithmic y-axis)
    # We use a marker to see individual points clearly, especially if some are 0
    plt.semilogy(x, probs, color='blue', marker='o', markersize=3, linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Styling
    plt.title("Spike Count Probability Distribution", fontsize=14)
    plt.xlabel("Spike Count Value (0-255)", fontsize=12)
    plt.ylabel("Probability (Log Scale)", fontsize=12)
    
    # Set x-limits to match the byte range
    plt.xlim(0, 255)
    plt.ylim(1e-10, 1e-0)

    # Add grid for easier reading of log values
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save the output
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to '{output_image}'")
    
    # Clean up memory
    plt.close()

if __name__ == "__main__":

    INPUT_FILE = "rodent_token_probabilities.npy" 
    OUTPUT_IMAGE = "rodent_token_probabilities.png"
    plot_probabilities(INPUT_FILE, OUTPUT_IMAGE)