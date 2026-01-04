#!/usr/bin/env python3
"""
Standalone script to test CUDA cuInit() directly with full logging.

This script attempts to initialize CUDA using the driver API and reports
detailed information about success or failure.
"""

import sys
import ctypes
import os

# Enable verbose CUDA logging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all TensorFlow logs

print("=" * 70)
print("CUDA cuInit() Test")
print("=" * 70)
print()

# Try to load CUDA driver library
cuda_lib = None
lib_names = [
    'libcuda.so.1',      # Linux
    'libcuda.so',        # Linux alternative
    'cuda.dll',          # Windows
    'nvcuda.dll',        # Windows alternative
]

for lib_name in lib_names:
    try:
        cuda_lib = ctypes.CDLL(lib_name)
        print(f"✓ Loaded CUDA library: {lib_name}")
        break
    except OSError:
        continue

if cuda_lib is None:
    print("✗ Failed to load CUDA driver library")
    print(f"  Tried: {lib_names}")
    sys.exit(1)

print()

# Define CUDA error codes (from cuda.h)
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_UNKNOWN = 999

error_names = {
    CUDA_SUCCESS: "CUDA_SUCCESS",
    CUDA_ERROR_INVALID_VALUE: "CUDA_ERROR_INVALID_VALUE",
    CUDA_ERROR_OUT_OF_MEMORY: "CUDA_ERROR_OUT_OF_MEMORY",
    CUDA_ERROR_NOT_INITIALIZED: "CUDA_ERROR_NOT_INITIALIZED",
    CUDA_ERROR_DEINITIALIZED: "CUDA_ERROR_DEINITIALIZED",
    CUDA_ERROR_NO_DEVICE: "CUDA_ERROR_NO_DEVICE",
    CUDA_ERROR_INVALID_DEVICE: "CUDA_ERROR_INVALID_DEVICE",
    CUDA_ERROR_INVALID_IMAGE: "CUDA_ERROR_INVALID_IMAGE",
    CUDA_ERROR_INVALID_CONTEXT: "CUDA_ERROR_INVALID_CONTEXT",
    CUDA_ERROR_INVALID_PTX: "CUDA_ERROR_INVALID_PTX",
    CUDA_ERROR_INVALID_HANDLE: "CUDA_ERROR_INVALID_HANDLE",
    CUDA_ERROR_UNKNOWN: "CUDA_ERROR_UNKNOWN",
}

# Get cuInit function
try:
    cuInit = cuda_lib.cuInit
    cuInit.argtypes = [ctypes.c_uint]
    cuInit.restype = ctypes.c_int
    print("✓ Found cuInit() function")
except AttributeError:
    print("✗ cuInit() function not found in library")
    sys.exit(1)

# Get cuGetErrorString function for better error messages
try:
    cuGetErrorString = cuda_lib.cuGetErrorString
    cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cuGetErrorString.restype = ctypes.c_int
    print("✓ Found cuGetErrorString() function")
    have_error_string = True
except AttributeError:
    print("⚠ cuGetErrorString() not available")
    have_error_string = False

print()
print("-" * 70)
print("Calling cuInit(0)...")
print("-" * 70)
print()

# Call cuInit
result = cuInit(0)

print(f"cuInit() returned: {result}")

if result == CUDA_SUCCESS:
    print(f"  Status: {error_names.get(result, 'UNKNOWN')} ({result})")
    print()
    print("✓✓✓ SUCCESS ✓✓✓")
    print()

    # Try to get device count
    try:
        cuDeviceGetCount = cuda_lib.cuDeviceGetCount
        cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuDeviceGetCount.restype = ctypes.c_int

        device_count = ctypes.c_int()
        result = cuDeviceGetCount(ctypes.byref(device_count))

        if result == CUDA_SUCCESS:
            print(f"Device count: {device_count.value}")

            # Get device properties
            for i in range(device_count.value):
                cuDeviceGet = cuda_lib.cuDeviceGet
                cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
                cuDeviceGet.restype = ctypes.c_int

                device = ctypes.c_int()
                result = cuDeviceGet(ctypes.byref(device), i)

                if result == CUDA_SUCCESS:
                    # Get device name
                    cuDeviceGetName = cuda_lib.cuDeviceGetName
                    cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
                    cuDeviceGetName.restype = ctypes.c_int

                    name_buffer = ctypes.create_string_buffer(256)
                    result = cuDeviceGetName(name_buffer, 256, device)

                    if result == CUDA_SUCCESS:
                        print(f"  Device {i}: {name_buffer.value.decode('utf-8')}")

                    # Get compute capability
                    cuDeviceGetAttribute = cuda_lib.cuDeviceGetAttribute
                    cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
                    cuDeviceGetAttribute.restype = ctypes.c_int

                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

                    major = ctypes.c_int()
                    minor = ctypes.c_int()

                    cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
                    cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)

                    print(f"    Compute capability: {major.value}.{minor.value}")
    except Exception as e:
        print(f"Note: Could not query device info: {e}")

    print()
    print("=" * 70)
    sys.exit(0)

else:
    print(f"  Status: {error_names.get(result, 'UNKNOWN')} ({result})")

    # Try to get error string
    if have_error_string:
        error_str = ctypes.c_char_p()
        err_result = cuGetErrorString(result, ctypes.byref(error_str))
        if err_result == CUDA_SUCCESS and error_str.value:
            print(f"  Error message: {error_str.value.decode('utf-8')}")

    print()
    print("✗✗✗ FAILED ✗✗✗")
    print()
    print("Debugging information:")
    print()

    # Check nvidia-smi
    print("Running nvidia-smi to check GPU status...")
    print("-" * 70)
    os.system("nvidia-smi 2>&1 | head -20")
    print("-" * 70)
    print()

    # Check device files
    print("Checking /dev/nvidia* permissions...")
    os.system("ls -la /dev/nvidia* 2>&1")
    print()

    # Check driver version
    print("Checking NVIDIA driver version...")
    os.system("cat /proc/driver/nvidia/version 2>&1")
    print()

    print("=" * 70)
    sys.exit(1)
