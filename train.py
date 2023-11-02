import torch

if not torch.cuda.is_available():
    print("CUDA is not enabled on your system. Please enable it to train the model.")
    print("Download the right version of pytorch here https://download.pytorch.org/whl/cu118")
    exit(1)

print(f"CUDA is enabled on this system.")
print(f"CUDA version: {torch.version.cuda}")
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")



