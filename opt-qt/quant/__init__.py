"""
OPT Model Quantization Framework

This package provides quantization support for OPT models,
including quantized linear layers, matrix multiplication, and
statistics collection mechanisms.
"""

from .quant_linear import QuantizedLinear
from .quant_matmul import QuantizedMatMul, MatMul
from .stat_manager import QuantStatManager, QuantStatistics
from .opt_wrapper import wrap_opt_model
from .model_wrapper import wrap_model_by_family
from .qwen_wrapper import switch_quantization_mode_all 
from .utils import (
    Round,
    Floor,
    load_config,
    save_scales,
    load_scales,
)

__version__ = "0.1.0"

__all__ = [
    "QuantizedLinear",
    "QuantizedMatMul",
    "MatMul",
    "QuantStatManager",
    "QuantStatistics",
    "wrap_opt_model",
    "Round",
    "Floor",
    "load_config",
    "save_scales",
    "load_scales",
    "wrap_model_by_family",
    "switch_quantization_mode_all",
]

