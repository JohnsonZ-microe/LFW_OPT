import torch
from typing import Dict, Any, List
import json
import sys
import os

def Systolic_HW_info():
    M = 64  # Tile rows
    N = 64  # Tile columns
    K = 1024  # Tile depth
    offchip_bandwidth = 64
    L1_cache_size = 128 * 1024
    maximum_thoughput = 256
    return M, N, K


def estimate_cache_access_latency(M: int, K: int, N: int, element_size=4, offchip_bandwidth=64):
    """Estimate transfer latency for [M, K] and [K, N] tensors from off-chip to on-chip."""
    a_size = M * K * element_size
    b_size = K * N * element_size
    total_size = a_size + b_size
    transfer_cycles = (total_size + offchip_bandwidth - 1) // offchip_bandwidth
    return {
        "a_size": a_size,
        "b_size": b_size,
        "total_size": total_size,
        "transfer_cycles": transfer_cycles
    }


def Systolic_proxy(
    M: int,
    N: int,
    K: int,
    x: torch.Tensor,
    w: torch.Tensor
):
    x_tile_num = x.size(0) // M
    w_tile_num = w.size(1) // N
    
    transfer_latency = estimate_cache_access_latency(M, K, N)
    compute_latency = M//16 * N//16 * K 
    compute_latency_ratio = compute_latency / (compute_latency + transfer_latency["transfer_cycles"])
    total_latency = transfer_latency["transfer_cycles"] + compute_latency
    external_memory_accesses = w_tile_num*x.size(0)*x.size(1) + w.size(0)*w.size(1) + x.size(0)*w.size(0)  # output stationary dataflow

    return {
        "total_latency": total_latency,
        "compute_latency_ratio": compute_latency_ratio,
        "external_memory_accesses": external_memory_accesses
    }
    