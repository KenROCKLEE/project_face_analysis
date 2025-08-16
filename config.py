import torch
import torchvision

def check_gpu():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU is available! Using GPU.")
    else:
        print("CUDA not available. Using CPU.")

if __name__ == "__main__":
    check_gpu()
