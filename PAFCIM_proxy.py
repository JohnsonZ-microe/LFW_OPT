import torch
from typing import Dict, Any
import json
import sys
import os

# /d:/HKUST/Chiplet_VLA/PAFCIM_proxy.py

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

# 示例用法（测试）
if __name__ == "__main__":
    # 生成一个示例 256-length 的 uint8 tensor（随机）
    t = torch.randint(0, 256, (256,), dtype=torch.uint8)
    res = analyze_e4m3_bits(t)
    print(res)
    print("Script file:", os.path.abspath(__file__))
    print("Result summary:")
    print(json.dumps(res, indent=2))

    # If you run the script by double-clicking or in an environment that closes the window,
    # pause so you can see the output.
    try:
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to exit...")
    except Exception:
        pass
    # print(f"starting {os.path.basename(__file__)}")