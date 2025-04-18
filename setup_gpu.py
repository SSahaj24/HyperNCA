import subprocess
import sys
import os
import platform

def check_gpu():
    """Check for CUDA-capable GPU and display information."""
    try:
        import torch
        print("\n=== GPU Information ===")
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"PyTorch version: {torch.__version__}")
            return True
        else:
            print("No CUDA-capable GPU detected.")
            print(f"PyTorch version: {torch.__version__}")
            return False
    except ImportError:
        print("PyTorch is not installed. Installing required packages...")
        return False

def install_packages():
    """Install required packages for GPU support."""
    system = platform.system()
    
    print("\n=== Installing Required Packages ===")
    
    if system == "Windows":
        # For Windows, need to use compatible CUDA version with PyTorch
        print("Installing PyTorch with CUDA support for Windows...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==1.11.0+cu113", "torchvision==0.12.0+cu113", "--extra-index-url", "https://download.pytorch.org/whl/cu113"])
    else:
        # For Linux/Mac
        print("Installing PyTorch with CUDA support...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==1.11.0", "torchvision==0.12.0"])
    
    # Install other required packages
    print("\nInstalling other dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def provide_instructions():
    """Provide instructions for running the GPU-accelerated code."""
    print("\n=== Instructions for Running GPU-Accelerated HyperNCA ===")
    print("1. Make sure your NVIDIA drivers are up to date")
    print("2. If you're running on Windows, ensure you have the Microsoft Visual C++ Redistributable installed")
    print("3. To run the training process:")
    print("   python train_NCA.py")
    print("   This will automatically use GPU acceleration if available")
    print("\n4. To control the number of GPUs used:")
    print("   python train_NCA.py --threads <num_gpus>")
    print("   This will limit the number of GPUs used for parallelization")

def main():
    print("===== HyperNCA GPU Setup =====")
    
    # Check if GPU is available
    has_gpu = check_gpu()
    
    # Install required packages
    install_packages()
    
    # Check again after installation
    if not has_gpu:
        has_gpu = check_gpu()
    
    # Provide instructions
    provide_instructions()
    
    if has_gpu:
        print("\n✅ Setup complete! Your system is ready to run HyperNCA with GPU acceleration.")
    else:
        print("\n⚠️ No GPU detected. HyperNCA will run on CPU only, which will be significantly slower.")
        print("If you have a compatible GPU, please make sure your NVIDIA drivers are installed correctly.")

if __name__ == "__main__":
    main() 