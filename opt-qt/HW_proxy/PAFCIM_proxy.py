import torch
from typing import Dict, Any, List
import json
import sys
import os

# /d:/HKUST/Chiplet_VLA/PAFCIM_proxy.py
def PAFCIM_HW_info():
    M = 64  # Tile rows
    N = 64  # Tile columns
    K = 1024  # Tile depth
    offchip_bandwidth = 64
    L1_cache_size = 128 * 1024
    maximum_thoughput = 256
    return M, N, K


def analyze_e4m3_bits(x: torch.Tensor) -> Dict[str, Any]:
    """
    分析 E4M3 (8-bit) 数组中的 sign 与 mantissa 的 1-bit 统计。
    输入:
      x: torch.Tensor，包含 256 个元素，元素应为无符号 8-bit 表示（0..255）。
         如果输入为整数型 Tensor，会被转为 torch.uint8；如果是浮点型会抛错。
    返回:
      dict 包含:
        - 'n_values': 元素数量（应为256）
        - 'sign': {'ones': int, 'proportion': float}
        - 'mantissa': {
              'total_bits': int,           # = n_values * 3
              'ones': int,
              'proportion': float,
              'per_bit': [ {'bit': 0/1/2, 'ones': int, 'proportion': float}, ... ]
          }
    约定：E4M3 在一个字节中的位分布按常见约定为
      [sign:1][exponent:4][mantissa:3]  -> sign 是最高位 (bit 7)，mantissa 是最低 3 位 (bits 0-2)。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("输入必须是 torch.Tensor")
    n = x.numel()
    if n != 256:
        raise ValueError(f"输入元素数量必须为256，当前为 {n}")
    if x.is_floating_point():
        raise TypeError("输入不应为浮点 tensor，应为整数表示的 8-bit 原始字节")
    # flatten and convert to uint8
    xb = x.flatten().to(dtype=torch.uint8).clone()

    # sign bit: bit 7
    sign_bits = (xb >> 7) & 1  # 0/1 tensor
    sign_ones = int(sign_bits.sum().item())
    sign_prop = sign_ones / n

    # mantissa bits: bits 0..2
    mant = xb & 0x07  # keep lowest 3 bits
    # total ones in mantissa across all values:
    # since mant has only 3 bits, sum of per-bit counts is popcount
    bit0 = (mant >> 0) & 1
    bit1 = (mant >> 1) & 1
    bit2 = (mant >> 2) & 1
    b0_ones = int(bit0.sum().item())
    b1_ones = int(bit1.sum().item())
    b2_ones = int(bit2.sum().item())
    total_mantissa_ones = b0_ones + b1_ones + b2_ones
    total_mantissa_bits = n * 3
    mantissa_prop = total_mantissa_ones / total_mantissa_bits

    per_bit = [
        {"bit": 0, "ones": b0_ones, "proportion": b0_ones / n},
        {"bit": 1, "ones": b1_ones, "proportion": b1_ones / n},
        {"bit": 2, "ones": b2_ones, "proportion": b2_ones / n},
    ]

    return {
        "n_values": n,
        "sign": {"ones": sign_ones, "proportion": sign_prop},
        "mantissa": {
            "total_bits": total_mantissa_bits,
            "ones": total_mantissa_ones,
            "proportion": mantissa_prop,
            "per_bit": per_bit,
        },
    }

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

def split_matrix_to_tiles(matrix: torch.Tensor, tile_rows: int, tile_cols: int) -> torch.Tensor:
    if matrix.dim() != 2:
        raise ValueError("矩阵必须是 2D")
    if matrix.size(1) != tile_cols:
        raise ValueError(f"矩阵列数应为 {tile_cols}，当前为 {matrix.size(1)}")
    if matrix.size(0) % tile_rows != 0:
        raise ValueError(f"矩阵行数 {matrix.size(0)} 不能被 tile_rows {tile_rows} 整除")
    # Reshape to [num_tiles, tile_rows, tile_cols] using tensor operations
    num_tiles = matrix.size(0) // tile_rows
    return matrix.reshape(num_tiles, tile_rows, tile_cols)

def PAFCIM_proxy(
    M: int,
    N: int,
    K: int,
    x: torch.Tensor,
    w: torch.Tensor
):
    x_tiles = split_matrix_to_tiles(x, M, K)
    w_tiles = split_matrix_to_tiles(w, N, K)
    x_tile_num = x.size(0) // M
    w_tile_num = w.size(1) // N

    transfer_latency = estimate_cache_access_latency(M, K, N)
    # Split each x_tile into [M, 32] submatrices (N_act pieces)
    # Split each w_tile into [32, 32] submatrices (N_w pieces)
    total_mantissa_ones_sum = 0

    if K % 32 != 0:
        raise ValueError("K must be divisible by 32 to split x into [M,32] submatrices")
    x_submatrices = torch.split(x_tiles, 32, dim=2)

    if K % 32 != 0 or N % 32 != 0:
        raise ValueError("K and N must be divisible by 32 to split w into [32,32] submatrices")
    
    N_w = (N // 32) * (K // 32)

    # Analyze each x_tile submatrix with analyze_e4m3_bits
    for x_sub in x_submatrices:
        # Flatten and convert to uint8 for analysis
        x_sub_flat = x_sub.flatten().to(dtype=torch.uint8)
        for i in range(0, x_sub_flat.numel(), 256):
            chunk = x_sub_flat[i:i+256]
            if chunk.numel() == 256:
                result = analyze_e4m3_bits(chunk)
                total_mantissa_ones_sum += result["mantissa"]["ones"]

    # Calculate total_latency based on total_mantissa_ones
    compute_latency = total_mantissa_ones_sum // N_w if N_w > 0 else 0
    compute_latency_ratio = compute_latency / (compute_latency + transfer_latency["transfer_cycles"])
    total_latency = transfer_latency["transfer_cycles"] + compute_latency
    external_memory_accesses = w_tile_num*x.size(0)*x.size(1) + w.size(0)*w.size(1) + x.size(0)*w.size(0)  # output stationary dataflow

    return external_memory_accesses, compute_latency_ratio, total_latency
