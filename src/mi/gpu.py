import torch


def check_gpu_availability() -> None:
    """Print the PyTorch version and report which accelerator, if any, torch can use.

    The branches are ordered most-specific-first and each backend is probed
    through its own `is_available()`, since a torch build only exposes a working
    backend for the hardware/driver it was compiled against.
    """
    print(f"PyTorch version {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA/ROCm GPU: {torch.cuda.get_device_name(0)}")
    elif torch.xpu.is_available():
        print(f"Intel GPU: {torch.xpu.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPU")
    else:
        print("Only CPU")


def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device