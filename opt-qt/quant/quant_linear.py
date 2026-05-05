"""
Quantized Linear Layer Implementation.

This module provides a quantized linear layer that supports:
- Multiple quantization modes (raw, scale_inspection, quant_forward)
- Configurable bit widths for activations, weights, and outputs
- Scale calibration and quantized inference
"""

import os
import pickle
from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .utils import Round, LINEAR_SHIFT_NUM


class QuantizedLinear(nn.Linear):
    """
    Quantized linear layer supporting multiple quantization modes.
    
    Modes:
        - 'raw': No quantization, standard linear layer
        - 'scale_inspection': Collect scale statistics for calibration
        - 'quant_forward': Quantized forward pass using pre-calibrated scales
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "raw",
        a_bit: int = 8,
        w_bit: int = 8,
        o_bit: int = 8,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        scale_root_str: str = "",
        outlier_ratio: float = 0.0,   # 新增，默认0表示不做outlier处理
        **kwargs
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input feature size
            out_features: Output feature size
            bias: Whether to use bias
            mode: Quantization mode ('raw', 'scale_inspection', 'quant_forward')
            a_bit: Activation quantization bits
            w_bit: Weight quantization bits
            o_bit: Output quantization bits
            d_bit: Digit size for quantization
            p: Parallelism parameter
            scale_root_str: Root directory for scale files
        """
        super().__init__(in_features, out_features, bias)
        
        # Quantization parameters
        self.mode = mode
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.o_bit = o_bit
        self.digit_size = d_bit
        self.parallelism = p
        
        # Quantization intervals (scales)
        self.w_interval: Optional[float] = None
        self.a_interval: Optional[float] = None
        self.o_interval: Optional[float] = None
        
        # Quantization maximum values
        self.w_qmax = 2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** (self.a_bit - 1)
        self.o_qmax = 2 ** (self.o_bit - 1)
        
        # Scale root directory
        self.scale_root_str = scale_root_str
        
        # Round function
        self.round = Round
        
        # Layer identification
        self.layer_name = ""
        self.layer_idx = 0
        
        self.outlier_ratio = outlier_ratio
        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None
    
    def set_layer_info(self, layer_name: str, layer_idx: int):
        """Set layer name and index for scale file management."""
        self.layer_name = layer_name
        self.layer_idx = layer_idx
    
    
    def _scale_file_paths(self):
        return (
            os.path.join(self.scale_root_str, f"{self.layer_name}_w_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_a_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_o_scale_{self.layer_idx}.p"),
        )

    def _scale_files_exist(self) -> bool:
        return all(os.path.exists(p) for p in self._scale_file_paths())

    def _resolve_calibration_action(self) -> str:
        if self._calibration_action_cache is not None:
            return self._calibration_action_cache

        p = str(self.calibration_policy).lower()
        if p == "auto":
            action = "reuse" if self._scale_files_exist() else "recalibrate"
        elif p in {"reuse", "recalibrate"}:
            action = p
        else:
            raise ValueError(
                f"Invalid calibration_policy={self.calibration_policy} "
                f"for {self.layer_name}_{self.layer_idx}"
            )

        self._calibration_action_cache = action
        return action


    def forward(
        self, 
        x: torch.Tensor,
        collect_stats: bool = False,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            collect_stats: Whether to collect statistics
            stat_collector: Statistics collector object
            
        Returns:
            Output tensor (quantized or not depending on mode)
        """
        # Use stored stat_manager if not provided
        if stat_collector is None and hasattr(self, '_stat_manager'):
            stat_collector = self._stat_manager
        
        if self.mode == 'raw':
            return F.linear(x, self.weight, self.bias)
        elif self.mode == "scale_inspection":
            return self.scale_inspection(x, stat_collector)
        elif self.mode == "quant_forward":
            return self.quant_forward(x, stat_collector)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
    
    def quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor."""
        return (w / self.w_interval).round_().clamp_(
            -self.w_qmax, self.w_qmax - 1
        )

    def quant_bias(self, b: torch.Tensor) -> torch.Tensor:
        """Quantize bias tensor."""
        biasfp32 = b.to(torch.float32)
        bias_sim = self.round(
            biasfp32 / (self.a_interval * self.w_interval) 
        )
        return bias_sim

    def quant_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor."""
        return (x / self.a_interval).round_().clamp_(
            -self.a_qmax, self.a_qmax - 1
        )
    
    def scale_inspection(
        self, 
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Inspect and collect scale statistics for calibration.
        
        w_scale and a_scale are from FP values.
        o_scale is from quantized output (to match quant_forward behavior).
        But we return FP output for numerical stability.
        """
        action = self._resolve_calibration_action()
        if action == "reuse":
            return F.linear(x, self.weight, self.bias)
        # Calculate weight and activation scales from FP values
        if self.outlier_ratio > 0.0:
            x_normal_mask = ~self._get_outlier_mask_1d(x, self.outlier_ratio)
            x_normal = x[x_normal_mask]
            if x_normal.numel() == 0:
                pass
            self.a_interval = (x_normal.abs().max() / (self.a_qmax - 0.5)).item()
            if self.a_interval == 0:
                self.a_interval = None
        
            w_normal_mask = ~self._get_outlier_mask_1d(self.weight, self.outlier_ratio)
            w_normal = self.weight[w_normal_mask]
            self.w_interval = (w_normal.abs().max() / (self.w_qmax - 0.5)).item()
        else:
            self.w_interval = (self.weight.abs().max() / (self.w_qmax - 0.5)).item()
            self.a_interval = (x.abs().max() / (self.a_qmax - 0.5)).item()
        
            # Simulate quantization to get o_scale (must match quant_forward)

        out = F.linear(x, self.weight, self.bias)
        
            # Calculate o_scale from quantized output
        self.o_interval = (out.abs().max() / (self.o_qmax - 0.5)).item()
        
            # Collect statistics if collector is provided
        if stat_collector is not None:
            stat_collector.collect_linear_stats(
                self.layer_name,
                self.layer_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval
            )
        
        # Return FP output for numerical stability during calibration
        return F.linear(x, self.weight, self.bias)
    
    def quant_forward(
        self, 
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        self._load_scales()

        if self.outlier_ratio > 0.0:
            return self._quant_forward_with_outlier(x, stat_collector)

        M0 = torch.tensor(
            self.w_interval * self.a_interval / self.o_interval,
            device=x.device,
            dtype=torch.float32,
        )
        M0 = self.round(M0 * LINEAR_SHIFT_NUM)

        w_sim = self.quant_weight(self.weight)
        x_sim = self.quant_input(x)

        if self.bias is not None:
            bias_sim = self.quant_bias(self.bias)
        else:
            bias_sim = None

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x_sim,
                self.a_bit,
                self.digit_size,
                self.parallelism
            )

        x_sim = x_sim.to(torch.float32)
        w_sim = w_sim.to(torch.float32)
        if bias_sim is not None:
            bias_sim = bias_sim.to(torch.float32)

        out_quant = F.linear(x_sim, w_sim, bias_sim)
        out_quant = out_quant.mul_(M0)
        out_quant = torch.div(
            out_quant,
            LINEAR_SHIFT_NUM,
            rounding_mode="floor",
        )

        out = out_quant.mul_(self.o_interval).to(x.dtype)

        x_ref = x
        if x_ref.dtype != self.weight.dtype:
            x_ref = x_ref.to(self.weight.dtype)

        if self.layer_name == "up_proj":
            if self.layer_idx == 1:
                pass

        out_ref = F.linear(x_ref, self.weight, self.bias).to(out.dtype)
        mse = ((out - out_ref) ** 2).mean().item()
        self.log_quant_error(
            mse*100/out.abs().mean().item(),  # 相对误差百分比
            layer_name=self.layer_name,
            layer_idx=self.layer_idx,
        )

        return out
    
    def _quant_forward_with_outlier(self, x, stat_collector=None):
        """带 outlier 保护的量化前向计算。"""

        if self.layer_name is "q_proj":
            pass

        x_outlier_mask = self._get_outlier_mask_1d(x, self.outlier_ratio)
        w_outlier_mask = self._get_outlier_mask_1d(self.weight, self.outlier_ratio)
        x_normal_mask = ~x_outlier_mask
        w_normal_mask = ~w_outlier_mask

        # === 分离 outlier 和 normal 部分（FP16）===
        x_fp = x * x_outlier_mask.to(torch.float32)        # A 的 outlier，保留 FP16
        x_normal_fp = x * x_normal_mask.to(torch.float32)  # A 的 normal，FP16（待量化）
        w_fp = self.weight * w_outlier_mask.to(torch.float32)        # B 的 outlier，保留 FP16
        w_normal_fp = self.weight * w_normal_mask.to(torch.float32)  # B 的 normal，FP16（待量化）
    
        # === 量化 normal 部分 ===
        M0 = torch.tensor(self.w_interval * self.a_interval / self.o_interval)
        M0 = self.round(M0 * LINEAR_SHIFT_NUM)

        M3 = torch.tensor(self.w_interval / self.o_interval)
        M3 = self.round(M3 * LINEAR_SHIFT_NUM)

        M2 = torch.tensor(self.a_interval / self.o_interval)
        M2 = self.round(M2 * LINEAR_SHIFT_NUM)

        M1 = torch.tensor(1.0 / self.o_interval)
        M1 = self.round(M1 * LINEAR_SHIFT_NUM)


        if self.bias is not None:
                bias_sim = self.quant_bias(self.bias)
        w_sim = self.quant_weight(w_normal_fp)
        x_sim = self.quant_input(x_normal_fp)
    
        x_sim_fp32 = x_sim.to(torch.float32)
        w_sim_fp32 = w_sim.to(torch.float32)
        if self.bias is not None:
            bias_sim = bias_sim.to(torch.float32) 
            bias = self.bias.to(torch.float32)
        else:
            bias = None
    
        
        out_quant = F.linear(x_sim_fp32, w_sim_fp32, None)
        out_quant = out_quant.mul_(M0)
        out_quant = torch.div(out_quant, LINEAR_SHIFT_NUM, rounding_mode='floor')

    
        out_fp_a = F.linear(x_fp, w_fp, None)
        a1   = torch.div((F.linear(x_sim_fp32, w_fp, None)).mul_(M2), LINEAR_SHIFT_NUM)       # outlier_x @ normal_W（FP精度）
        a2   = torch.div((F.linear(x_fp, w_sim_fp32, None)).mul_(M3), LINEAR_SHIFT_NUM)

        a0_ref = F.linear(x_normal_fp, w_normal_fp, None)
        a1_ref = F.linear(x_normal_fp, w_fp, None)
        a2_ref = F.linear(x_fp, w_normal_fp, None)

        mse_a0 = ((out_quant*self.o_interval - a0_ref) ** 2).mean().item()
        mse_a1 = ((a1.floor()*self.o_interval - a1_ref) ** 2).mean().item()
        mse_a2 = ((a2.floor()*self.o_interval - a2_ref) ** 2).mean().item()

        out_fp = a1 + a2
        out_fp = out_fp.floor()

        out = out_quant + out_fp

        out = (out.mul_(self.o_interval).to(x.dtype) + out_fp_a).to(self.weight.dtype)
        """

        x_tst = (x_sim_fp32*self.a_interval + x_fp)
        w_tst = (w_sim_fp32*self.w_interval + w_fp)
        out = F.linear(x_tst, w_tst, bias).to(self.weight.dtype)
        """
        x_ref = x.to(self.weight.dtype)
        out_ref = F.linear(x_ref, self.weight, self.bias)
        mse = ((out - out_ref) ** 2).mean().item()
        self.log_quant_error(mse, layer_name=self.layer_name, layer_idx=self.layer_idx)

        if torch.isnan(out).max():
            pass

        return out
    
    def _get_outlier_mask_1d(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        返回 bool mask，True 表示是 outlier（保留 FP16）。
        离群值定义为绝对值最大的元素，数量约占总元素数的 ratio（至少1个，最多 numel-1 个）。
        当所有元素绝对值相等时，返回全 False。
        """
        if ratio <= 0.0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    
        numel = tensor.numel()
        if numel == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    
        # 至少选1个，最多选 numel-1 个，确保正常部分非空
        k = max(1, min(int(numel * ratio), numel - 1))
    
        flat_abs = tensor.abs().flatten()
        # 获取第 k 大的值
        threshold = torch.topk(flat_abs, k).values.min()
    
        # 处理阈值等于最小值的情况
        min_val = flat_abs.min()
        if threshold == min_val:
            # 只选严格大于最小值的元素，避免全选
            outlier_mask_flat = flat_abs > min_val
        else:
            outlier_mask_flat = flat_abs >= threshold
    
        # 如果离群数量为0（如全等值），返回全False
        if outlier_mask_flat.sum() == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    
        # 恢复原始形状
        return outlier_mask_flat.view(tensor.shape)


    def _get_weight_outlier_mask(self, ratio: float) -> torch.Tensor:
        """
        按列（输出通道）计算权重 outlier mask。
        离群列定义为列最大绝对值最大的那些列，数量约占列数的 ratio（至少1列，最多 in_features-1 列）。
        当所有列最大值相等时，返回全 False。
        """
        if ratio <= 0.0:
            return torch.zeros(self.weight.shape, dtype=torch.bool, device=self.weight.device)
    
        col_max = self.weight.abs().max(dim=0).values  # [in_features]
        num_cols = col_max.numel()
        if num_cols == 0:
            return torch.zeros(self.weight.shape, dtype=torch.bool, device=self.weight.device)
    
        # 至少选1列，最多选 num_cols-1 列
        k = max(1, min(int(num_cols * ratio), num_cols - 1))
    
        threshold = torch.topk(col_max, k).values.min()
        min_val = col_max.min()
    
        if threshold == min_val:
            outlier_cols = col_max > min_val
        else:
            outlier_cols = col_max >= threshold
    
        # 如果没有离群列，返回全False
        if outlier_cols.sum() == 0:
            return torch.zeros(self.weight.shape, dtype=torch.bool, device=self.weight.device)
    
        # 扩展成与 weight 相同形状的 mask
        return outlier_cols.unsqueeze(0).expand_as(self.weight)

    def _load_scales(self):
        """Load quantization scales from files."""
        # Construct scale file paths
        w_scale_file = os.path.join(
            self.scale_root_str, 
            f"{self.layer_name}_w_scale_{self.layer_idx}.p"
        )
        a_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_a_scale_{self.layer_idx}.p"
        )
        o_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_o_scale_{self.layer_idx}.p"
        )
        
        # Load scales
        with open(w_scale_file, 'rb') as f:
            self.w_interval = pickle.load(f)
        
        with open(a_scale_file, 'rb') as f:
            self.a_interval = pickle.load(f)
        
        with open(o_scale_file, 'rb') as f:
            self.o_interval = pickle.load(f)
    
    def save_scales(self):
        """Save current scales to files."""
        if self.w_interval is None or self.a_interval is None or self.o_interval is None:
            raise ValueError("Scales not computed yet. Run scale_inspection first.")
        
        os.makedirs(self.scale_root_str, exist_ok=True)
        
        # Save weight scale
        w_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_w_scale_{self.layer_idx}.p"
        )
        with open(w_scale_file, 'wb') as f:
            pickle.dump(self.w_interval, f)
        
        # Save activation scale
        a_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_a_scale_{self.layer_idx}.p"
        )
        with open(a_scale_file, 'wb') as f:
            pickle.dump(self.a_interval, f)
        
        # Save output scale
        o_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_o_scale_{self.layer_idx}.p"
        )
        with open(o_scale_file, 'wb') as f:
            pickle.dump(self.o_interval, f)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'mode={self.mode}, '
            f'a_bit={self.a_bit}, '
            f'w_bit={self.w_bit}, '
            f'o_bit={self.o_bit}'
        )


    def log_quant_error(self, value: float, layer_name: str = None, layer_idx: int = None):
        import json as _json
        import os as _os
        layer_name = layer_name if layer_name is not None else self.layer_name
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx
        json_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "quant-error", "error.jsonl"
        )
        _os.makedirs(_os.path.dirname(json_path), exist_ok=True)
        entry = {"layer_name": layer_name, "layer_idx": layer_idx, "value": float(value)}
        with open(json_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
