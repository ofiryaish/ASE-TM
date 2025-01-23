import torch
import time


def gpu_keep_alive(duration_seconds=3600):
    """
    Keeps the GPU active by running a dummy computation for a specified duration using PyTorch.

    Args:
        duration_seconds (int): Total time (in seconds) to keep the GPU busy.
    """
    # Ensure a GPU is available
    if not torch.cuda.is_available():
        print("No GPU detected. Please ensure your system has a GPU.")
        return

    # Create some large tensors on the GPU
    device = torch.device("cuda:0")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Start the keep-alive loop
    start_time = time.time()
    print(f"GPU keep-alive started for {duration_seconds} seconds.")

    while time.time() - start_time < duration_seconds:
        # Perform a matrix multiplication to keep the GPU active
        result = torch.matmul(a, b)
        # Use the result to ensure computation is not optimized away
        result.mean().item()
        # Add a small delay to reduce power usage
        time.sleep(0.01)

    print("GPU keep-alive completed.")


# Run the keep-alive script for 1 hour (3600 seconds)
if __name__ == "__main__":
    gpu_keep_alive(3600*20)
