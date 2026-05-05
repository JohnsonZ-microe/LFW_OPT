"""
Quantization modules for RoBERTa model.

This module provides quantized linear and matrix multiplication layers
for model compression and acceleration.
"""

import pickle
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple, List, Any
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure this module can be imported consistently as both "Quant.Roberta_quant" and "Roberta_quant".
# This prevents class-identity mismatches (e.g., MatMul) when different import paths are used.
if __name__ == "Quant.Roberta_quant":
    sys.modules.setdefault("Roberta_quant", sys.modules[__name__])
elif __name__ == "Roberta_quant":
    sys.modules.setdefault("Quant.Roberta_quant", sys.modules[__name__])


# ============================================================================
# Configuration Constants
# ============================================================================

DEFAULT_QUANT_SCALE_DIR = "/home//zyzhao//Desktop//HPCA2026_FOCUS//Quant//quant_scale//"

# Default quantization configuration (will be updated with actual quant_scale_dir)
DEFAULT_QUANT_CONFIG = {
    "scale_root_str": DEFAULT_QUANT_SCALE_DIR
}

# Quantization bit configurations
# Format: (activation_bits, weight_bits, output_bits)
QUANT_BIT_CONFIGS = {
    "query": (8, 4, 6),
    "key": (8, 4, 8),
    "value": (8, 8, 8),
    "attention_output": (6, 4, 8),
    "fc1": (8, 8, 8),
    "fc2": (8, 8, 8),
    "classifier": (8, 8, 8),
    "qk_mul": (6, 8, 10),  # A_bit, B_bit, O_bit
    "pv_mul": (10, 8, 6),  # A_bit, B_bit, O_bit
}

# Shift numbers for quantization
LINEAR_SHIFT_NUM = 2 ** 16
MATMUL_SHIFT_NUM = 2 ** 20

# Default quantization parameters
DEFAULT_DIGIT_SIZE = 4
DEFAULT_PARALLELISM = 4

# Default quantization bit width configuration (defined after DEFAULT_DIGIT_SIZE and DEFAULT_PARALLELISM)
# This can be overridden by passing a custom config dict
DEFAULT_QUANT_BIT_CONFIG = {
    # Scale inspection modules
    "scale_insp_q": {"a_bit": 8, "w_bit": 4, "o_bit": 6},
    "scale_insp_k": {"a_bit": 8, "w_bit": 4, "o_bit": 8},
    "scale_insp_v": {"a_bit": 8, "w_bit": 8, "o_bit": 8},
    "scale_insp_aod": {"a_bit": 6, "w_bit": 4, "o_bit": 8},
    "scale_insp_id": {"a_bit": 8, "w_bit": 8, "o_bit": 8},
    "scale_insp_od": {"a_bit": 8, "w_bit": 8, "o_bit": 8},
    "scale_insp_cd": {"a_bit": 8, "w_bit": 8, "o_bit": 8},
    "scale_insp_qk": {"A_bit": 6, "B_bit": 8, "O_bit": 10},
    "scale_insp_PV": {"A_bit": 10, "B_bit": 8, "O_bit": 6},
    # Quantized forward modules
    "qlinear_q": {"a_bit": 8, "w_bit": 4, "o_bit": 6, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_k": {"a_bit": 8, "w_bit": 4, "o_bit": 8, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_v": {"a_bit": 8, "w_bit": 8, "o_bit": 8, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_attn_out": {"a_bit": 6, "w_bit": 4, "o_bit": 8, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_FC1": {"a_bit": 8, "w_bit": 8, "o_bit": 8, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_FC2": {"a_bit": 8, "w_bit": 8, "o_bit": 8, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_LMhead": {"a_bit": 8, "w_bit": 8, "o_bit": 8, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_QKmul": {"A_bit": 8, "B_bit": 8, "O_bit": 10, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
    "qlinear_PVmul": {"A_bit": 10, "B_bit": 8, "O_bit": 6, "d_bit": DEFAULT_DIGIT_SIZE, "p": DEFAULT_PARALLELISM},
}


# ============================================================================
# STE (Straight-Through Estimator) Functions
# ============================================================================

class RoundSTE(Function):
    """Straight-through estimator for rounding operation."""
    
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        """Round inputs in forward pass."""
        return torch.round(inputs)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradients through unchanged."""
        return grad_output.clone()


class FloorSTE(Function):
    """Straight-through estimator for floor operation."""
    
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        """Floor inputs in forward pass."""
        return torch.floor(inputs)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradients through unchanged."""
        return grad_output.clone()


# Apply functions
Round = RoundSTE.apply
Floor = FloorSTE.apply


# ============================================================================
# Base Modules
# ============================================================================

class MatMul(nn.Module):
    """Simple matrix multiplication module."""
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication A @ B."""
        return A @ B


# #region agent log
def _agent_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any], run_id: str = "pre-fix") -> None:
    """Append one NDJSON debug log line for debug-mode analysis."""
    try:
        payload = {
            "id": f"log_{int(time.time() * 1000)}_{os.getpid()}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open("/home/zyzhao/Desktop/HPCA2026_FOCUS/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion agent log


# ============================================================================
# Quantized Linear Layer
# ============================================================================

class QuantizedLinear(nn.Linear):
    """
    Quantized linear layer supporting multiple quantization modes.
    
    Modes:
        - 'raw': No quantization, standard linear layer
        - 'quant_forward': Quantized forward pass using pre-calibrated scales
        - 'scale_inspection': Inspect and collect scale statistics
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "raw",
        a_bit: int = 9,
        w_bit: int = 9,
        o_bit: int = 9,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input feature size
            out_features: Output feature size
            bias: Whether to use bias
            mode: Quantization mode ('raw', 'quant_forward', 'scale_inspection')
            a_bit: Activation quantization bits
            w_bit: Weight quantization bits
            o_bit: Output quantization bits
            d_bit: Digit size for quantization
            p: Parallelism parameter
            **kwargs: Additional arguments (must include 'scale_root_str')
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
        
        # Scale root directory for loading calibration data
        self.scale_root_str = kwargs.get('scale_root_str', '')
        
        # Statistics and tracking
        self.layer_idx = 0
        self.round = Round
        
        # Legacy attributes (kept for compatibility)
        self.n_calibration_step = 2
        self.raw_input = None
        self.raw_out = None
        self.metric = None
        self.next_nodes = []
        self.model_stat = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        name: List[str], 
        model_stat: Optional[Dict] = None, 
        layer_ind: int = 0
    ) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            name: List of name identifiers for statistics collection
            model_stat: Dictionary of model statistics collectors
            layer_ind: Layer index
            
        Returns:
            Quantized output tensor
        """
        if self.mode == 'raw':
            return F.linear(x, self.weight, self.bias)
        elif self.mode == "quant_forward":
            return self.quant_forward(x, name, model_stat=model_stat, layer_ind=layer_ind)
        elif self.mode == "scale_inspection":
            return self.scale_inspection(x, name, model_stat=model_stat, layer_ind=layer_ind)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
    
    def quant_weight_bias(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize weights and bias."""
        w_sim = self.round(self.weight / self.w_interval).clamp_(
            -self.w_qmax, self.w_qmax - 1
        )
        
        if self.bias is not None:
            bias_sim = self.round(
                self.bias / (self.a_interval * self.w_interval)
            )
            return w_sim, bias_sim
        else:
            return w_sim, None
    
    def quant_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor."""
        return (x / self.a_interval).round_().clamp_(
            -self.a_qmax, self.a_qmax - 1
        )
    
    def scale_inspection(
        self, 
        x: torch.Tensor, 
        name: List[str], 
        model_stat: Optional[Dict] = None, 
        layer_ind: int = 0
    ) -> torch.Tensor:
        """
        Inspect and collect scale statistics.
        
        This mode calculates quantization scales dynamically and collects
        statistics for calibration.
        """
        # Calculate weight interval
        self.w_interval = (self.weight.abs().max() / (self.w_qmax - 0.5))
        w_interval_key = name[0] + str(layer_ind)
        if model_stat and w_interval_key in model_stat:
            self.w_interval = model_stat[w_interval_key].statistic_probe(self.w_interval)
        
        # Calculate activation interval
        self.a_interval = (x.abs().max() / (self.a_qmax - 0.5))
        a_interval_key = name[1] + str(layer_ind)
        if model_stat and a_interval_key in model_stat:
            self.a_interval = model_stat[a_interval_key].statistic_probe(self.a_interval)
        
        # Quantize and compute
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, w_sim, bias_sim)
        out = out * self.w_interval * self.a_interval
        
        # Calculate output interval
        self.o_interval = (out.abs().max() / (self.o_qmax - 0.5))
        out_interval_key = name[2] + str(layer_ind)
        if model_stat and out_interval_key in model_stat:
            self.o_interval = model_stat[out_interval_key].statistic_probe(self.o_interval)
        
        return out
    
    def quant_forward(
        self, 
        x: torch.Tensor, 
        name: List[str], 
        model_stat: Optional[Dict] = None, 
        layer_ind: int = 0
    ) -> torch.Tensor:
        """
        Quantized forward pass using pre-calibrated scales.
        
        Loads quantization scales from files and performs quantized computation.
        """
        # Load quantization intervals from files
        w_interval_file = f"{self.scale_root_str}{name[0]}{layer_ind}.p"
        with open(w_interval_file, 'rb') as f:
            self.w_interval = pickle.load(f)
        
        a_interval_file = f"{self.scale_root_str}{name[1]}{layer_ind}.p"
        with open(a_interval_file, 'rb') as f:
            self.a_interval = pickle.load(f)
        
        out_interval_file = f"{self.scale_root_str}{name[2]}{layer_ind}.p"
        with open(out_interval_file, 'rb') as f:
            self.o_interval = pickle.load(f)
        
        # Calculate scaling factor M0
        M0 = torch.tensor(self.w_interval * self.a_interval / self.o_interval)
        M0 = self.round(M0 * LINEAR_SHIFT_NUM)
        
        # Quantize weights, bias, and input
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        
        # Collect statistics if model_stat is provided
        if model_stat:
            x_quant_key = name[4] + str(layer_ind)
            if x_quant_key in model_stat:
                model_stat[x_quant_key].statistic_probe([
                    x_sim, self.a_bit, self.digit_size, self.parallelism
                ])
        
        # Perform quantized linear operation
        out_quant = F.linear(x_sim, w_sim, bias_sim)
        out_quant = out_quant.mul_(M0)
        out_quant = torch.div(out_quant, LINEAR_SHIFT_NUM)
        out = out_quant.mul_(self.o_interval)
        
        return out


# ============================================================================
# Quantized Matrix Multiplication Layer
# ============================================================================

class QuantizedMatMul(nn.Module):
    """
    Quantized matrix multiplication layer.
    
    Supports quantization for attention mechanism matrix multiplications.
    """
    
    def __init__(
        self,
        mode: str = "raw",
        A_bit: Optional[int] = None,
        B_bit: Optional[int] = None,
        O_bit: Optional[int] = None,
        scale_root_str: Optional[str] = None,
        d_bit: Optional[int] = None,
        p: Optional[int] = None
    ):
        """
        Initialize quantized matrix multiplication layer.
        
        Args:
            mode: Quantization mode
            A_bit: Bits for first matrix (A)
            B_bit: Bits for second matrix (B)
            O_bit: Bits for output
            scale_root_str: Root directory for scale files
            d_bit: Digit size
            p: Parallelism parameter
        """
        super().__init__()
        self.mode = mode
        self.A_bit = A_bit
        self.B_bit = B_bit
        self.O_bit = O_bit
        self.digit_size = d_bit
        self.parallelism = p
        self.scale_root_str = scale_root_str
        self.round = Round
        
        # Quantization intervals
        self.A_interval: Optional[float] = None
        self.B_interval: Optional[float] = None
        self.O_interval: Optional[float] = None
        
        # Quantization maximum values
        if A_bit:
            self.A_qmax = 2 ** (A_bit - 1)
        if B_bit:
            self.B_qmax = 2 ** (B_bit - 1)
        if O_bit:
            self.O_qmax = 2 ** (O_bit - 1)
        
        # Legacy attributes
        self.raw_input = None
        self.raw_out = None
    
    def forward(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        name: List[str], 
        model_stat: Optional[Dict] = None, 
        layer_ind: int = 0
    ) -> torch.Tensor:
        """Forward pass with quantization."""
        if self.mode == 'raw':
            return A @ B
        elif self.mode == "quant_forward":
            return self.quant_forward(A, B, name, model_stat=model_stat, layer_ind=layer_ind)
        elif self.mode == "scale_inspection":
            return self.scale_inspection(A, B, name, model_stat=model_stat, layer_ind=layer_ind)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
    
    def quant_input(self, x: torch.Tensor, interval: float, qmax: int) -> torch.Tensor:
        """Quantize input tensor."""
        return (x / interval).round_().clamp_(-qmax, qmax - 1)
    
    def scale_inspection(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        name: List[str], 
        model_stat: Optional[Dict] = None, 
        layer_ind: int = 0
    ) -> torch.Tensor:
        """Inspect and collect scale statistics for matrix multiplication."""
        # Calculate intervals
        self.A_interval = (A.abs().max() / (self.A_qmax - 0.5))
        self.B_interval = (B.abs().max() / (self.B_qmax - 0.5))
        
        # Collect statistics if available
        if model_stat:
            A_interval_key = name[0] + str(layer_ind)
            if A_interval_key in model_stat:
                self.A_interval = model_stat[A_interval_key].statistic_probe(self.A_interval)
            
            B_interval_key = name[1] + str(layer_ind)
            if B_interval_key in model_stat:
                self.B_interval = model_stat[B_interval_key].statistic_probe(self.B_interval)
        
        # Quantize and compute
        A_sim = self.quant_input(A, self.A_interval, self.A_qmax)
        B_sim = self.quant_input(B, self.B_interval, self.B_qmax)
        out = (A_sim @ B_sim) * self.A_interval * self.B_interval
        
        # Calculate output interval
        self.O_interval = (out.abs().max() / (self.O_qmax - 0.5))
        if model_stat:
            O_interval_key = name[2] + str(layer_ind)
            if O_interval_key in model_stat:
                self.O_interval = model_stat[O_interval_key].statistic_probe(self.O_interval)
        
        return out
    
    def quant_forward(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        name: List[str], 
        model_stat: Optional[Dict] = None, 
        layer_ind: int = 0
    ) -> torch.Tensor:
        """Quantized forward pass using pre-calibrated scales."""
        # Load intervals from files
        A_interval_file = f"{self.scale_root_str}{name[0]}{layer_ind}.p"
        with open(A_interval_file, 'rb') as f:
            self.A_interval = pickle.load(f)
        
        B_interval_file = f"{self.scale_root_str}{name[1]}{layer_ind}.p"
        with open(B_interval_file, 'rb') as f:
            self.B_interval = pickle.load(f)
        
        O_interval_file = f"{self.scale_root_str}{name[2]}{layer_ind}.p"
        with open(O_interval_file, 'rb') as f:
            self.O_interval = pickle.load(f)
        
        # Calculate scaling factor
        M0 = torch.tensor(self.A_interval * self.B_interval / self.O_interval)
        M0 = self.round(M0 * MATMUL_SHIFT_NUM)
        
        # Quantize inputs
        A_sim = self.quant_input(A, self.A_interval, self.A_qmax)
        B_sim = self.quant_input(B, self.B_interval, self.B_qmax)
        
        # Collect statistics if available
        if model_stat:
            A_quant_key = name[3] + str(layer_ind)
            if A_quant_key in model_stat:
                model_stat[A_quant_key].statistic_probe([
                    A_sim, self.A_bit, self.digit_size, self.parallelism
                ])
        
        # Perform quantized matrix multiplication
        out_quant = A_sim @ B_sim
        out_quant = out_quant.mul_(M0)
        out_quant = torch.div(out_quant, MATMUL_SHIFT_NUM)
        out = out_quant.mul_(self.O_interval)
        
        return out


# ============================================================================
# Module Wrapping Functions
# ============================================================================

def _get_module_type_mapping(linear_layer_quant: str) -> Dict[str, str]:
    """
    Get module type mapping based on quantization mode.
    
    Args:
        linear_layer_quant: Quantization mode ('qlinear' or 'scale_insp')
        
    Returns:
        Dictionary mapping module names to module types
    """
    if linear_layer_quant == "qlinear":
        return {
            "query": "qlinear_q",
            "key": "qlinear_k",
            "value": "qlinear_v",
            "qk_mul": "qlinear_QKmul",
            "PV_mul": "qlinear_PVmul",
            "attention.output.dense": "qlinear_attn_out",
            "intermediate.dense": "qlinear_FC1",
            "output.dense": "qlinear_FC2",
            "classifier.dense": "qlinear_LMhead",
            "out_proj": "no_quant"
        }
    else:
        return {
            "query": "scale_insp_q",
            "key": "scale_insp_k",
            "value": "scale_insp_v",
            "qk_mul": "scale_insp_qk",
            "PV_mul": "scale_insp_PV",
            "attention.output.dense": "scale_insp_aod",
            "intermediate.dense": "scale_insp_id",
            "output.dense": "scale_insp_od",
            "classifier.dense": "scale_insp_cd",
            "out_proj": "no_quant"
        }


def _create_quantized_module(
    module_type: str, 
    in_features: Optional[int] = None, 
    out_features: Optional[int] = None,
    quant_scale_dir: Optional[str] = None,
    quant_bit_config: Optional[Dict[str, Dict[str, int]]] = None,
    layer_index: Optional[int] = None
) -> nn.Module:
    """
    Create a quantized module based on type.
    
    Args:
        module_type: Type of module to create
        in_features: Input features (for linear layers)
        out_features: Output features (for linear layers)
        quant_scale_dir: Directory path for quantization scale files
        quant_bit_config: Dictionary of bit width configurations for each module type
        layer_index: Layer index for per-layer configuration (optional)
        
    Returns:
        Quantized module instance
    """
    # Use provided quant_scale_dir or default
    if quant_scale_dir is None:
        quant_scale_dir = DEFAULT_QUANT_SCALE_DIR
    
    # Use provided bit config or default
    if quant_bit_config is None:
        quant_bit_config = DEFAULT_QUANT_BIT_CONFIG
    
    # Get bit config for this module type
    # Support both flat config and layer-wise config
    if layer_index is not None:
        # Try layer-wise config first: config[layer_idx][module_type]
        layer_key = f"{module_type}_layer{layer_index}"
        if layer_key in quant_bit_config:
            bit_config = quant_bit_config[layer_key]
        elif layer_index in quant_bit_config and module_type in quant_bit_config[layer_index]:
            bit_config = quant_bit_config[layer_index][module_type]
        else:
            # Fall back to module type only
            bit_config = quant_bit_config.get(module_type, {})
    else:
        # Use module type only
        bit_config = quant_bit_config.get(module_type, {})
    
    # Update kwargs with actual scale directory
    kwargs = {"scale_root_str": quant_scale_dir}
    
    # Scale inspection modules
    if module_type == "scale_insp_q":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 4), 
            o_bit=bit_config.get("o_bit", 6),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_k":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 4), 
            o_bit=bit_config.get("o_bit", 8),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_v":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_aod":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 6), 
            w_bit=bit_config.get("w_bit", 4), 
            o_bit=bit_config.get("o_bit", 8),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_id":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_od":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_cd":
        return QuantizedLinear(
            mode="scale_inspection", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "scale_insp_qk":
        return QuantizedMatMul(
            mode="scale_inspection",
            scale_root_str=quant_scale_dir,
            A_bit=bit_config.get("A_bit", 6), 
            B_bit=bit_config.get("B_bit", 8), 
            O_bit=bit_config.get("O_bit", 10)
        )
    elif module_type == "scale_insp_PV":
        return QuantizedMatMul(
            mode="scale_inspection",
            scale_root_str=quant_scale_dir,
            A_bit=bit_config.get("A_bit", 10), 
            B_bit=bit_config.get("B_bit", 8), 
            O_bit=bit_config.get("O_bit", 6)
        )
    
    # Quantized forward modules
    elif "qlinear_q" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 4), 
            o_bit=bit_config.get("o_bit", 6),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif "qlinear_k" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 4), 
            o_bit=bit_config.get("o_bit", 8),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif "qlinear_v" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif "qlinear_attn_out" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 6), 
            w_bit=bit_config.get("w_bit", 4), 
            o_bit=bit_config.get("o_bit", 8),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif "qlinear_FC1" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif "qlinear_FC2" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif "qlinear_LMhead" in module_type:
        return QuantizedLinear(
            mode="quant_forward", 
            a_bit=bit_config.get("a_bit", 8), 
            w_bit=bit_config.get("w_bit", 8), 
            o_bit=bit_config.get("o_bit", 8),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM),
            in_features=in_features, out_features=out_features, **kwargs
        )
    elif module_type == "qlinear_QKmul":
        return QuantizedMatMul(
            mode="quant_forward",
            scale_root_str=quant_scale_dir,
            A_bit=bit_config.get("A_bit", 8), 
            B_bit=bit_config.get("B_bit", 8), 
            O_bit=bit_config.get("O_bit", 10),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM)
        )
    elif "qlinear_PVmul" in module_type:
        return QuantizedMatMul(
            mode="quant_forward",
            scale_root_str=quant_scale_dir,
            A_bit=bit_config.get("A_bit", 10), 
            B_bit=bit_config.get("B_bit", 8), 
            O_bit=bit_config.get("O_bit", 6),
            d_bit=bit_config.get("d_bit", DEFAULT_DIGIT_SIZE), 
            p=bit_config.get("p", DEFAULT_PARALLELISM)
        )
    
    # No quantization
    elif module_type == "no_quant":
        return QuantizedLinear(
            mode="raw",
            in_features=in_features, out_features=out_features, **kwargs
        )
    
    else:
        raise ValueError(f"Unknown module type: {module_type}")


def _replace_linear_layer(
    module: nn.Linear,
    module_name: str,
    father_name: str,
    father_module: nn.Module,
    module_types: Dict[str, str],
    wrapped_modules: Dict[str, nn.Module],
    quant_scale_dir: Optional[str] = None,
    quant_bit_config: Optional[Dict[str, Dict[str, int]]] = None,
    layer_index: Optional[int] = None
):
    """
    Replace a linear layer with quantized version.
    
    Args:
        module: Original linear layer
        module_name: Full name of the module
        father_name: Name of parent module
        father_module: Parent module object
        module_types: Mapping of module patterns to types
        wrapped_modules: Dictionary to store wrapped modules
        quant_scale_dir: Directory path for quantization scale files
        quant_bit_config: Bit configuration dictionary
        layer_index: Layer index for per-layer configuration
    """
    idx = module_name.rfind('.')
    idx = idx + 1 if idx != 0 else idx
    module_suffix = module_name[idx:]
    
    if module_suffix == 'dense':
        # Handle different dense layer types
        if father_name[father_name.rfind('.') - 1:father_name.rfind('.')] == 'n':
            # attention.output.dense
            module_type_key = module_name[idx - 17:]
            new_module = _create_quantized_module(
                module_types[module_type_key],
                module.in_features, module.out_features,
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
            new_module.weight.data = module.weight.data
            new_module.bias = module.bias
            wrapped_modules[module_name] = new_module
            setattr(father_module, module_suffix, new_module)
        elif father_name[father_name.rfind('.') + 1:] == 'intermediate':
            # intermediate.dense
            module_type_key = module_name[idx - 13:]
            new_module = _create_quantized_module(
                module_types[module_type_key],
                module.in_features, module.out_features,
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
            new_module.weight = module.weight
            new_module.bias = module.bias
            wrapped_modules[module_name] = new_module
            setattr(father_module, module_suffix, new_module)
        elif father_name[father_name.rfind('.') + 1:] == 'classifier':
            # classifier.dense
            module_type_key = module_name[idx - 11:]
            new_module = _create_quantized_module(
                module_types[module_type_key],
                module.in_features, module.out_features,
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
            new_module.weight.data = module.weight.data
            new_module.bias = module.bias
            wrapped_modules[module_name] = new_module
            setattr(father_module, module_suffix, new_module)
        else:
            # output.dense
            module_type_key = module_name[idx - 7:]
            new_module = _create_quantized_module(
                module_types[module_type_key],
                module.in_features, module.out_features,
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
            new_module.weight.data = module.weight.data
            new_module.bias = module.bias
            wrapped_modules[module_name] = new_module
            setattr(father_module, module_suffix, new_module)
    else:
        # Handle query, key, value, and out_proj layers
        if module_suffix == 'out_proj':
            # out_proj is set to no_quant (no quantization applied)
            # Note: Original code had this commented out, keeping it for compatibility
            pass
        else:
            # query, key, value layers
            new_module = _create_quantized_module(
                module_types[module_suffix],
                module.in_features, module.out_features,
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
            new_module.weight.data = module.weight.data
            new_module.bias = module.bias
            wrapped_modules[module_name] = new_module
            setattr(father_module, module_suffix, new_module)


def wrap_modules_in_net(
    net: nn.Module, 
    linear_layer_quant: str = "qlinear", 
    enable_quantization: bool = False,
    quant_scale_dir: Optional[str] = None,
    quant_bit_config: Optional[Dict[str, Dict[str, int]]] = None
) -> Dict[str, nn.Module]:
    """
    Wrap model modules with quantized versions.
    
    Args:
        net: PyTorch model to wrap
        linear_layer_quant: Quantization mode ('qlinear' or 'scale_insp')
        enable_quantization: Whether to enable quantization (default: False)
        quant_scale_dir: Directory path for quantization scale files
        quant_bit_config: Dictionary of bit width configurations for each module type
        
    Returns:
        Dictionary of wrapped modules
    """
    # #region agent log
    _agent_log(
        "H1",
        "Quant/Roberta_quant.py:wrap_modules_in_net",
        "enter wrap_modules_in_net",
        {
            "linear_layer_quant": linear_layer_quant,
            "enable_quantization": enable_quantization,
            "quant_scale_dir": quant_scale_dir,
            "MatMul_class_module": getattr(MatMul, "__module__", None),
            "MatMul_class_id": id(MatMul),
        },
        run_id="post-fix",
    )
    # #endregion agent log
    if not enable_quantization:
        print("Quantization disabled. Using original model.")
        return {}
    
    module_types = _get_module_type_mapping(linear_layer_quant)
    module_dict = {}
    wrapped_modules = {}
    
    # Build module dictionary
    for name, module in net.named_modules():
        module_dict[name] = module
    
    # Replace modules with quantized versions
    for name, module in net.named_modules():
        idx = name.rfind('.')
        father_name = name[:idx] if idx != -1 else ""
        
        if father_name and father_name not in module_dict:
            raise RuntimeError(f"Parent module {father_name} not found")
        
        father_module = module_dict[father_name] if father_name else net
        
        # Extract layer index from module name (e.g., "roberta.encoder.layer.0.attention...")
        layer_index = None
        if 'encoder.layer.' in name:
            try:
                # Extract layer number from path like "roberta.encoder.layer.0.attention..."
                parts = name.split('.')
                layer_idx_pos = parts.index('layer') + 1
                layer_index = int(parts[layer_idx_pos])
            except (ValueError, IndexError):
                pass
        
        if isinstance(module, nn.Linear):
            _replace_linear_layer(
                module, name, father_name, father_module,
                module_types, wrapped_modules, 
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
        elif isinstance(module, MatMul) or type(module).__name__ == "MatMul":
            # Replace matrix multiplication layers
            idx = idx + 1 if idx != 0 else idx
            module_suffix = name[idx:]
            # #region agent log
            _agent_log(
                "H1",
                "Quant/Roberta_quant.py:wrap_modules_in_net",
                "MatMul candidate encountered",
                {
                    "name": name,
                    "module_suffix": module_suffix,
                    "type_name": type(module).__name__,
                    "type_module": getattr(type(module), "__module__", None),
                    "type_id": id(type(module)),
                    "isinstance_MatMul": isinstance(module, MatMul),
                    "mapping_has_key": module_suffix in module_types,
                    "mapped_type": module_types.get(module_suffix, None),
                },
                run_id="post-fix",
            )
            # #endregion agent log
            new_module = _create_quantized_module(
                module_types[module_suffix], 
                quant_scale_dir=quant_scale_dir,
                quant_bit_config=quant_bit_config,
                layer_index=layer_index
            )
            wrapped_modules[name] = new_module
            setattr(father_module, module_suffix, new_module)
            # #region agent log
            _agent_log(
                "H2",
                "Quant/Roberta_quant.py:wrap_modules_in_net",
                "MatMul replaced with quantized module",
                {
                    "name": name,
                    "module_suffix": module_suffix,
                    "new_type_name": type(new_module).__name__,
                    "new_type_module": getattr(type(new_module), "__module__", None),
                    "attr_type_name": type(getattr(father_module, module_suffix)).__name__,
                    "attr_type_module": getattr(type(getattr(father_module, module_suffix)), "__module__", None),
                },
                run_id="post-fix",
            )
            # #endregion agent log
    
    print("Completed net wrap.")
    return wrapped_modules


# Backward compatibility aliases
MyQuantLinear = QuantizedLinear
MyQuantMatMul = QuantizedMatMul
get_module = _create_quantized_module
quant_kwargs = DEFAULT_QUANT_CONFIG
quant_scale_dir = DEFAULT_QUANT_SCALE_DIR
QUANT_SCALE_DIR = DEFAULT_QUANT_SCALE_DIR  # For backward compatibility
