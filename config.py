import torch
import torchvision

def check_setup():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available. Using CPU.")

if __name__ == "__main__":
    check_setup()