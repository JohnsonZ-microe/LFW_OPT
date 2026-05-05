"""
Statistics Manager for Quantization Calibration.

This module provides classes for collecting and managing quantization statistics
during the calibration phase using PyTorch hooks.
"""

import os
import pickle
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from collections import defaultdict


class QuantStatistics:
    """
    Statistics collector for a single layer.
    
    Collects scale information for activations, weights, and outputs
    during calibration.
    """
    
    def __init__(self, layer_name: str, layer_idx: int):
        """
        Initialize statistics collector.
        
        Args:
            layer_name: Name of the layer (e.g., 'q_proj', 'k_proj')
            layer_idx: Index of the layer in the model
        """
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        
        # Scale statistics
        self.w_scales: List[float] = []
        self.a_scales: List[float] = []
        self.o_scales: List[float] = []
        
        # For matmul layers
        self.A_scales: List[float] = []
        self.B_scales: List[float] = []
        
        # Sample count
        self.sample_count = 0
    
    def collect_linear_stats(
        self, 
        w_scale: float, 
        a_scale: float, 
        o_scale: float
    ):
        """
        Collect statistics for linear layer.
        
        Args:
            w_scale: Weight scale
            a_scale: Activation scale
            o_scale: Output scale
        """
        self.w_scales.append(w_scale)
        self.a_scales.append(a_scale)
        self.o_scales.append(o_scale)
        self.sample_count += 1
    
    def collect_matmul_stats(
        self,
        A_scale: float,
        B_scale: float,
        O_scale: float
    ):
        """
        Collect statistics for matrix multiplication.
        
        Args:
            A_scale: First input scale
            B_scale: Second input scale
            O_scale: Output scale
        """
        if A_scale is not None:
            self.A_scales.append(A_scale)
        if B_scale is not None:
            self.B_scales.append(B_scale)
        self.o_scales.append(O_scale)
        self.sample_count += 1
        if self.layer_idx == 0:
            if self.layer_name =="qk_matmul":
                pass


    def get_final_scales(self) -> Dict[str, float]:
        """
        Compute final scales from collected statistics.
        
        Uses maximum value across all samples for robustness.
        
        Returns:
            Dictionary of final scales
        """
        scales = {}
        
        if self.w_scales:
            scales['w_scale'] = max(self.w_scales)
            scales['a_scale'] = max(self.a_scales)
            scales['o_scale'] = max(self.o_scales)
        
        if self.A_scales:
            scales['A_scale'] = max(self.A_scales)
            scales['B_scale'] = max(self.B_scales)
            # For matmul, O_scale is stored in o_scales list
            if self.o_scales:
                scales['O_scale'] = max(self.o_scales)
        
        return scales
    
    def reset(self):
        """Reset all collected statistics."""
        self.w_scales.clear()
        self.a_scales.clear()
        self.o_scales.clear()
        self.A_scales.clear()
        self.B_scales.clear()
        self.sample_count = 0


class QuantStatManager:
    """
    Global statistics manager for quantization calibration.
    
    Manages statistics collection for all layers in the model using
    PyTorch hooks.
    """
    
    def __init__(self, scale_dir: str):
        """
        Initialize statistics manager.
        
        Args:
            scale_dir: Directory to save/load scales
        """
        self.scale_dir = scale_dir
        self.stats: Dict[str, QuantStatistics] = {}
        self.hooks: List[Any] = []
        
        # Create scale directory if it doesn't exist
        os.makedirs(scale_dir, exist_ok=True)

        self.total_zero_count = 0
        self.total_element_count = 0
        self.total_bit_count = 0
        self.total_0bit_count = 0
        self.total_sparsebit_count = 0
        self.total_amplitude_zero_bits_total = 0
    
    def register_layer(self, layer_name: str, layer_idx: int):
        """
        Register a layer for statistics collection.
        
        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer
        """
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.stats[key] = QuantStatistics(layer_name, layer_idx)
    
    def collect_linear_stats(
        self,
        layer_name: str,
        layer_idx: int,
        w_scale: float,
        a_scale: float,
        o_scale: float
    ):
        """Collect statistics for linear layer."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.register_layer(layer_name, layer_idx)
        self.stats[key].collect_linear_stats(w_scale, a_scale, o_scale)
    
    def collect_global_sparsity(self, total_num: int, abs_less_th: int, total_bits: int, zero_bits_total: int, sparse_bits_total: int, amplitude_zero_bits_total: int):
        """
        Collect global sparsity statistics.
    
        Args:
            zero_count: Number of zero elements in this layer
            total_count: Total number of elements in this layer
        """
        self.total_zero_count += abs_less_th
        self.total_element_count += total_num
        self.total_bit_count += total_bits
        self.total_0bit_count += zero_bits_total
        self.total_sparsebit_count += sparse_bits_total
        self.total_amplitude_zero_bits_total += amplitude_zero_bits_total

    def collect_matmul_stats(
        self,
        layer_name: str,
        layer_idx: int,
        A_scale: float,
        B_scale: float,
        O_scale: float
    ):
        """Collect statistics for matrix multiplication."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.register_layer(layer_name, layer_idx)
        self.stats[key].collect_matmul_stats(A_scale, B_scale, O_scale)
    
    def collect_quant_activation(
        self,
        layer_name: str,
        layer_idx: int,
        activation: torch.Tensor,
        bit_width: int,
        digit_size: int,
        parallelism: int
    ):
        """
        Collect quantized activation statistics (for analysis).
        
        This is optional and used for analyzing quantization effects.
        """
        sparse = self.compute_sparse_stats(bit_width, activation, 0)
        self.collect_global_sparsity(*sparse)
    
    def compute_sparse_stats(self, n, tensor, th):

        # 1. 元素总数
        total_num = tensor.numel()

        # 2. 绝对值小于 th 的元素个数
        abs_less_th = (torch.abs(tensor) <= th).sum().item()

        # 3. 总比特数
        total_bits = total_num * n

        # 空张量快速返回
        if total_num == 0:
            return total_num, abs_less_th, total_bits, 0, 0, 0

        # 将 float16 转换为 int32（值均为整数，安全）
        int_tensor = tensor.to(torch.int32)

        # n 位掩码
        mask = (1 << n) - 1

        # 获取补码的低 n 位（无符号表示）
        us_tensor = int_tensor & mask

        # 正数掩码（包括 0）
        pos_mask = int_tensor >= 0

        # --- 构造原码（符号-绝对值表示） ---
        # 幅度部分的掩码（n-1 位）
        amp_mask = (1 << (n - 1)) - 1
        # 绝对值（int32）
        abs_val = torch.abs(int_tensor)
        # 负数的原码：符号位为 1，幅度为 abs_val 的低 n-1 位
        neg_orig = (1 << (n - 1)) | (abs_val & amp_mask)
        # 正数/零的原码就是补码本身（低 n 位）
        orig_tensor = torch.where(pos_mask, us_tensor, neg_orig)

        # 初始化累加器
        zero_bits_total = 0
        sparse_bits_total = 0
        amplitude_zero_bits_total = 0

        # 逐位统计
        for i in range(n):
            # 从补码取第 i 位
            bit = (us_tensor >> i) & 1

            # 补码中的 0 总数
            zero_bits_total += (1 - bit).sum().item()

            # 稀疏比特（正数取 !bit，负数取 bit）
            sparse_contrib = torch.where(pos_mask, 1 - bit, bit)
            sparse_bits_total += sparse_contrib.sum().item()

            # 原码中的 0 总数
            orig_bit = (orig_tensor >> i) & 1
            amplitude_zero_bits_total += (1 - orig_bit).sum().item()

        return (total_num, abs_less_th, total_bits, zero_bits_total, sparse_bits_total, amplitude_zero_bits_total)

    def save_all_scales(self):
        """Save all collected scales to files."""
        print(f"Saving scales to {self.scale_dir}...")
        
        for key, stat in self.stats.items():
            scales = stat.get_final_scales()
            
            # Save each scale type
            for scale_type, scale_value in scales.items():
                filename = f"{stat.layer_name}_{scale_type}_{stat.layer_idx}.p"
                filepath = os.path.join(self.scale_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(scale_value, f)
                
                print(f"  Saved {filename}: {scale_value:.6f}")
        
        print(f"Total scales saved: {len(self.stats)} layers")
    
    def load_all_scales(self) -> Dict[str, Dict[str, float]]:
        """
        Load all scales from files.
        
        Returns:
            Dictionary mapping layer keys to their scales
        """
        loaded_scales = {}
        
        for key, stat in self.stats.items():
            scales = {}
            
            # Try to load each scale type
            for scale_type in ['w_scale', 'a_scale', 'o_scale', 'A_scale', 'B_scale', 'O_scale']:
                filename = f"{stat.layer_name}_{scale_type}_{stat.layer_idx}.p"
                filepath = os.path.join(self.scale_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        scales[scale_type] = pickle.load(f)
            
            if scales:
                loaded_scales[key] = scales
        
        return loaded_scales
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected statistics.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'total_layers': len(self.stats),
            'total_samples': sum(stat.sample_count for stat in self.stats.values()),
            'layers': {}
        }
        
        for key, stat in self.stats.items():
            summary['layers'][key] = {
                'layer_name': stat.layer_name,
                'layer_idx': stat.layer_idx,
                'sample_count': stat.sample_count,
                'scales': stat.get_final_scales()
            }
        
        return summary
    
    def print_summary(self):
        """Print summary of collected statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("Quantization Statistics Summary")
        print("="*80)
        print(f"Total layers: {summary['total_layers']}")
        print(f"Total samples: {summary['total_samples']}")
        print("\nPer-layer statistics:")
        print("-"*80)
        
        for key, info in summary['layers'].items():
            print(f"\n{key}:")
            print(f"  Samples: {info['sample_count']}")
            print(f"  Scales:")
            for scale_name, scale_value in info['scales'].items():
                print(f"    {scale_name}: {scale_value:.6f}")
        
        print("="*80 + "\n")
    
    def reset_all(self):
        """Reset all statistics."""
        for stat in self.stats.values():
            stat.reset()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

