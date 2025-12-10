#!/usr/bin/env python3
"""
Test CUDA initialization through PyTorch with maximum verbosity.

This script enables all PyTorch/CUDA logging to see exactly what fails
during GPU initialization.
"""

import sys
import os

print("=" * 70)
print("PyTorch CUDA Initialization Test (Verbose)")
print("=" * 70)
print()

# Enable maximum verbosity BEFORE importing PyTorch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'          # Synchronous CUDA launches
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'   # Show C++ stack traces
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  # Detailed distributed debug

print("Environment variables set:")
print(f"  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
print(f"  TORCH_SHOW_CPP_STACKTRACES={os.environ.get('TORCH_SHOW_CPP_STACKTRACES')}")
print()
print("-" * 70)
print("Importing PyTorch...")
print("-" * 70)
print()

try:
    import torch
    print()
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Built with CUDA: {torch.cuda.is_available()}")
    print()
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    print()
    print("Install PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio")
    sys.exit(1)

print("-" * 70)
print("PyTorch Build Info:")
print("-" * 70)
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda if torch.version.cuda else 'N/A'}")
print(f"  cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print(f"  cuDNN enabled: {torch.backends.cudnn.enabled if torch.cuda.is_available() else 'N/A'}")

# Check if PyTorch was built with CUDA
if not torch.cuda.is_available():
    print()
    print("⚠ CUDA is NOT available to PyTorch")
    print()
    print("This could mean:")
    print("  1. PyTorch was built without CUDA support (CPU-only)")
    print("  2. CUDA_VISIBLE_DEVICES is set to hide GPUs")
    print("  3. CUDA driver initialization (cuInit) failed")
    print("  4. No NVIDIA GPUs present")
    print()

    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    if cuda_visible == '':
        print("  → GPU is hidden by CUDA_VISIBLE_DEVICES=''")

    print()
    print("✗✗✗ CUDA NOT AVAILABLE ✗✗✗")
    sys.exit(1)

print()
print("-" * 70)
print("Enumerating GPU devices...")
print("-" * 70)
print()

try:
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")
    print()

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")

        # Check compute capability
        if props.major >= 12:
            print(f"  ⚠ WARNING: Compute capability {props.major}.{props.minor} is very new")
            print(f"    Pre-compiled kernels may not be available")
            print(f"    JIT compilation may be required")
        elif props.major >= 10:
            print(f"  ⚠ NOTE: Compute capability {props.major}.{props.minor} is newer than most builds")

        print()

    # Set default device
    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    print(f"Current device set to: {current_device}")
    print()

except Exception as e:
    print(f"✗ Exception while enumerating devices: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

print("-" * 70)
print("Testing CUDA memory allocation...")
print("-" * 70)
print()

try:
    # Try to allocate a small tensor
    test_tensor = torch.zeros(10, 10, device='cuda')
    print(f"✓ Allocated test tensor on GPU: shape {test_tensor.shape}")
    print(f"  Device: {test_tensor.device}")
    del test_tensor
    torch.cuda.empty_cache()
    print("✓ Tensor deallocated successfully")
    print()

except Exception as e:
    print(f"✗ Memory allocation failed: {e}")
    error_str = str(e)

    if 'CUDA out of memory' in error_str:
        print()
        print("DIAGNOSIS: Out of GPU memory")
        print("  The GPU doesn't have enough free memory.")
    elif 'no kernel image' in error_str.lower():
        print()
        print("DIAGNOSIS: No compatible kernel image")
        print("  PyTorch doesn't have pre-compiled kernels for your GPU.")
        print("  Your compute capability may be too new.")
    elif 'CUDA error' in error_str:
        print()
        print("DIAGNOSIS: CUDA runtime error")
        print("  There was an error in the CUDA runtime.")

    import traceback
    print()
    print("Full traceback:")
    traceback.print_exc()
    print()
    sys.exit(1)

print("-" * 70)
print("Testing simple GPU computation...")
print("-" * 70)
print()

try:
    # Create tensors on GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')

    print(f"Created tensors: a={a.shape}, b={b.shape}")
    print(f"  Device: {a.device}")
    print()

    # Matrix multiplication
    print("Performing matrix multiplication (a @ b)...")
    c = torch.matmul(a, b)

    # Force synchronization to catch any async errors
    torch.cuda.synchronize()

    print(f"✓ Result shape: {c.shape}")
    print(f"  Sample values: {c[0, :5].cpu().numpy()}")
    print()

    # Memory info
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"GPU Memory:")
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved: {reserved:.2f} MB")
    print()

    # Clean up
    del a, b, c
    torch.cuda.empty_cache()

    print("✓✓✓ GPU COMPUTATION SUCCESSFUL ✓✓✓")
    print()
    print("=" * 70)
    sys.exit(0)

except RuntimeError as e:
    print(f"✗ GPU computation failed: {e}")
    print()
    error_str = str(e)

    if 'no kernel image is available' in error_str.lower():
        print("DIAGNOSIS: No kernel image available for computation")
        print("  PyTorch was not built with support for your GPU's compute capability.")
        print()
        print("  Your options:")
        print("    1. Use CPU-only mode:")
        print("       Set CUDA_VISIBLE_DEVICES='' before importing torch")
        print()
        print("    2. Build PyTorch from source with support for your GPU:")
        print("       https://github.com/pytorch/pytorch#from-source")
        print()
        print("    3. Wait for a PyTorch release with support for compute capability 12.0")
        print()

    elif 'CUDA error: invalid device function' in error_str:
        print("DIAGNOSIS: Invalid device function")
        print("  The GPU doesn't support the requested operation.")
        print("  Your compute capability may be too new or too old.")
        print()

    elif 'CUDA error: no kernel image' in error_str:
        print("DIAGNOSIS: No kernel image")
        print("  Similar to 'no kernel image available' - compute capability mismatch.")
        print()

    elif 'CUDA out of memory' in error_str:
        print("DIAGNOSIS: Out of GPU memory")
        print("  Not enough free GPU memory for the operation.")
        print()
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"  Allocated: {allocated:.2f} MB")
        print(f"  Reserved: {reserved:.2f} MB")
        print()

    else:
        print("DIAGNOSIS: Unknown CUDA error")
        print("  Check the full traceback below for details.")
        print()

    import traceback
    print("Full traceback:")
    traceback.print_exc()
    print()

except Exception as e:
    print(f"✗ Unexpected error: {type(e).__name__}")
    print(f"  {e}")
    print()
    import traceback
    traceback.print_exc()
    print()

print("=" * 70)
sys.exit(1)
