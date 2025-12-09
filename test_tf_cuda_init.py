#!/usr/bin/env python3
"""
Test CUDA initialization through TensorFlow with maximum verbosity.

This script enables all TensorFlow/CUDA logging to see exactly what fails
during GPU initialization.
"""

import sys
import os

print("=" * 70)
print("TensorFlow CUDA Initialization Test (Verbose)")
print("=" * 70)
print()

# Enable maximum verbosity BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'          # Show all logs (INFO, WARNING, ERROR)
os.environ['TF_CPP_VMODULE'] = 'cuda_diagnostics=10,cuda_executor=10,cuda_platform=10,gpu_device=10'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'          # Synchronous CUDA launches
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'         # Disable oneDNN for clarity

print("Environment variables set:")
print(f"  TF_CPP_MIN_LOG_LEVEL={os.environ['TF_CPP_MIN_LOG_LEVEL']}")
print(f"  TF_CPP_VMODULE={os.environ['TF_CPP_VMODULE']}")
print(f"  CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}")
print()
print("-" * 70)
print("Importing TensorFlow...")
print("-" * 70)
print()

try:
    import tensorflow as tf
    print()
    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ Built with CUDA: {tf.test.is_built_with_cuda()}")
    print()
except ImportError as e:
    print(f"✗ Failed to import TensorFlow: {e}")
    sys.exit(1)

print("-" * 70)
print("TensorFlow Build Info:")
print("-" * 70)
build_info = tf.sysconfig.get_build_info()
for key, value in build_info.items():
    print(f"  {key}: {value}")
print()

print("-" * 70)
print("Attempting to list physical GPU devices...")
print("-" * 70)
print()

try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Found {len(gpus)} GPU(s)")

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {details}")
            except Exception as e:
                print(f"    Could not get details: {e}")
        print()
        print("✓ GPU devices listed successfully")
    else:
        print("⚠ No GPUs found")
        print()
        print("This could mean:")
        print("  1. CUDA_VISIBLE_DEVICES is set to hide GPUs")
        print("  2. cuInit() failed during TensorFlow initialization")
        print("  3. No NVIDIA GPUs present")

except Exception as e:
    print(f"✗ Exception while listing GPUs: {e}")
    import traceback
    traceback.print_exc()

print()
print("-" * 70)
print("Attempting simple GPU operation...")
print("-" * 70)
print()

try:
    # Try to create a simple tensor on GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"Result of GPU matrix multiplication:\n{c.numpy()}")
        print()
        print("✓✓✓ GPU OPERATION SUCCESSFUL ✓✓✓")
        print()
        sys.exit(0)

except Exception as e:
    print(f"✗ GPU operation failed: {e}")
    print()
    error_str = str(e)

    if 'CUDA_ERROR_INVALID_PTX' in error_str:
        print("DIAGNOSIS: CUDA_ERROR_INVALID_PTX")
        print("  This means the GPU's compute capability is not supported.")
        print("  Your GPU is too new for this TensorFlow version.")
        print()
        print("  Solution: Use CPU-only mode")
        print("    os.environ['CUDA_VISIBLE_DEVICES'] = ''")

    elif 'CUDA_ERROR_INVALID_HANDLE' in error_str:
        print("DIAGNOSIS: CUDA_ERROR_INVALID_HANDLE")
        print("  This often follows INVALID_PTX errors.")
        print("  Your GPU's compute capability is not supported.")
        print()
        print("  Solution: Use CPU-only mode")
        print("    os.environ['CUDA_VISIBLE_DEVICES'] = ''")

    elif 'CUDA_ERROR_UNKNOWN' in error_str:
        print("DIAGNOSIS: CUDA_ERROR_UNKNOWN")
        print("  This is a generic CUDA error during cuInit().")
        print()
        print("  Possible causes:")
        print("    1. Driver/CUDA version mismatch")
        print("    2. GPU not properly initialized")
        print("    3. Permissions issue with /dev/nvidia*")
        print()
        print("  Try:")
        print("    - sudo nvidia-smi -pm 1  (enable persistence mode)")
        print("    - Reboot")
        print("    - Check: ls -la /dev/nvidia*")

    elif 'No GPU devices available' in error_str or 'device:GPU:0' in error_str:
        print("DIAGNOSIS: No GPU available to TensorFlow")
        print("  TensorFlow initialized but can't see/use the GPU.")
        print()
        print("  This likely means cuInit() or device enumeration failed.")

    import traceback
    print()
    print("Full traceback:")
    traceback.print_exc()

print()
print("=" * 70)
sys.exit(1)
