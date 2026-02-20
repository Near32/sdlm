#!/usr/bin/env python3
"""
Comprehensive AMD Ryzen AI Max Performance Benchmark
Tests various aspects of GPU compute performance on gfx1151
"""

import os
import torch
import time
import numpy as np
import warnings
from contextlib import contextmanager

# Configure environment for gfx1151
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1151"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"
os.environ["PYTORCH_DISABLE_FLASH_ATTENTION"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure attention backends
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

@contextmanager
def timer(description):
    """Context manager for timing operations"""
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")

def print_system_info():
    """Print system and GPU information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print(f"CPU count: {torch.get_num_threads()}")
    print()

def test_basic_operations():
    """Test basic tensor operations"""
    print("=" * 60)
    print("BASIC TENSOR OPERATIONS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different sizes
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices:")
        
        # CPU test
        a_cpu = torch.randn(size, size, dtype=torch.float32)
        b_cpu = torch.randn(size, size, dtype=torch.float32)
        
        with timer(f"  CPU matmul ({size}x{size})"):
            c_cpu = torch.matmul(a_cpu, b_cpu)
        
        if torch.cuda.is_available():
            # GPU test
            a_gpu = a_cpu.to(device)
            b_gpu = b_cpu.to(device)
            
            with timer(f"  GPU matmul ({size}x{size})"):
                c_gpu = torch.matmul(a_gpu, b_gpu)
                torch.cuda.synchronize()
            
            # Verify correctness
            diff = torch.mean(torch.abs(c_cpu - c_gpu.cpu())).item()
            print(f"  Difference CPU vs GPU: {diff:.2e}")
            
            # Calculate TFLOPS
            ops = 2 * size**3  # Matrix multiplication operations
            gpu_time = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_time
            
            tflops = ops / gpu_time / 1e12
            print(f"  GPU Performance: {tflops:.2f} TFLOPS")

def test_memory_bandwidth():
    """Test memory bandwidth"""
    print("=" * 60)
    print("MEMORY BANDWIDTH TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tests")
        return
    
    device = torch.device("cuda")
    
    # Test different data sizes (in MB)
    sizes_mb = [10, 100, 500, 1000, 2000]
    
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        
        print(f"\nTesting {size_mb} MB transfer:")
        
        # Host to Device
        data_cpu = torch.randn(num_elements, dtype=torch.float32)
        
        start_time = time.time()
        data_gpu = data_cpu.to(device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start_time
        
        h2d_bandwidth = (size_mb / h2d_time) / 1024  # GB/s
        print(f"  Host to Device: {h2d_bandwidth:.2f} GB/s")
        
        # Device to Host
        start_time = time.time()
        data_back = data_gpu.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start_time
        
        d2h_bandwidth = (size_mb / d2h_time) / 1024  # GB/s
        print(f"  Device to Host: {d2h_bandwidth:.2f} GB/s")
        
        # Device to Device (copy)
        start_time = time.time()
        data_copy = data_gpu.clone()
        torch.cuda.synchronize()
        d2d_time = time.time() - start_time
        
        d2d_bandwidth = (size_mb / d2d_time) / 1024  # GB/s
        print(f"  Device to Device: {d2d_bandwidth:.2f} GB/s")

def test_mixed_precision():
    """Test mixed precision performance"""
    print("=" * 60)
    print("MIXED PRECISION TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision tests")
        return
    
    device = torch.device("cuda")
    size = 1000
    
    print(f"Testing {size}x{size} matrix operations:")
    
    # FP32 test
    a_fp32 = torch.randn(size, size, dtype=torch.float32, device=device)
    b_fp32 = torch.randn(size, size, dtype=torch.float32, device=device)
    
    with timer("  FP32 matmul"):
        c_fp32 = torch.matmul(a_fp32, b_fp32)
        torch.cuda.synchronize()
    
    # FP16 test
    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()
    
    with timer("  FP16 matmul"):
        c_fp16 = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
    
    # BF16 test (if supported)
    try:
        a_bf16 = a_fp32.bfloat16()
        b_bf16 = b_fp32.bfloat16()
        
        with timer("  BF16 matmul"):
            c_bf16 = torch.matmul(a_bf16, b_bf16)
            torch.cuda.synchronize()
    except:
        print("  BF16 not supported")

def test_conv_operations():
    """Test convolution operations"""
    print("=" * 60)
    print("CONVOLUTION OPERATIONS")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping convolution tests")
        return
    
    device = torch.device("cuda")
    
    # Different convolution configurations
    configs = [
        (1, 64, 224, 224, 64, 3, 1),   # (batch, in_ch, H, W, out_ch, kernel, stride)
        (1, 128, 112, 112, 128, 3, 1),
        (1, 256, 56, 56, 256, 3, 1),
        (1, 512, 28, 28, 512, 3, 1),
    ]
    
    for batch, in_ch, h, w, out_ch, kernel, stride in configs:
        print(f"\nConv2d: {in_ch}→{out_ch}, {h}x{w}, k={kernel}, s={stride}")
        
        # Create input and conv layer
        x = torch.randn(batch, in_ch, h, w, device=device)
        conv = torch.nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel//2).to(device)
        
        # Forward pass
        with timer(f"  Forward pass"):
            y = conv(x)
            torch.cuda.synchronize()
        
        print(f"  Input shape: {list(x.shape)}")
        print(f"  Output shape: {list(y.shape)}")
        print(f"  Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def test_transformer_attention():
    """Test transformer attention mechanisms"""
    print("=" * 60)
    print("TRANSFORMER ATTENTION")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping attention tests")
        return
    
    device = torch.device("cuda")
    
    # Attention configurations
    configs = [
        (1, 512, 8, 64),    # (batch, seq_len, heads, head_dim)
        (1, 1024, 12, 64),
        (1, 2048, 16, 64),
    ]
    
    for batch, seq_len, num_heads, head_dim in configs:
        print(f"\nAttention: batch={batch}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
        
        # Create attention inputs
        hidden_size = num_heads * head_dim
        x = torch.randn(batch, seq_len, hidden_size, device=device)
        
        # Multi-head attention layer
        attention = torch.nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        ).to(device)
        
        with timer(f"  Multi-head attention"):
            attn_output, _ = attention(x, x, x)
            torch.cuda.synchronize()
        
        print(f"  Input shape: {list(x.shape)}")
        print(f"  Output shape: {list(attn_output.shape)}")
        print(f"  Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def test_model_inference():
    """Test actual model inference if transformers is available"""
    print("=" * 60)
    print("MODEL INFERENCE TEST")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping model tests")
            return
        
        device = torch.device("cuda")
        model_name = "distilbert-base-uncased"
        
        print(f"Loading {model_name}...")
        
        # Load model with eager attention
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Test inference
        text = "This is a test sentence for benchmarking transformer inference performance."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()
        
        # Benchmark
        num_runs = 10
        print(f"Running {num_runs} inference iterations...")
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model(**inputs)
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Throughput: {1/avg_time:.2f} inferences/second")
        print(f"Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    except ImportError:
        print("Transformers library not available, skipping model tests")
    except Exception as e:
        print(f"Model test failed: {e}")

def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("AMD Ryzen AI Max EVO-X2 Performance Benchmark")
    print("=" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_system_info()
    test_basic_operations()
    test_memory_bandwidth()
    test_mixed_precision()
    test_conv_operations()
    test_transformer_attention()
    test_model_inference()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"Final memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    run_all_benchmarks()
