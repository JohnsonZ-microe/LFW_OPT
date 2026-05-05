"""
Utility functions for quantization.

This module provides helper functions for:
- STE (Straight-Through Estimator) operations
- Configuration loading
- Scale saving/loading
- Logging utilities
"""

import os
import pickle
import yaml
import torch
from torch.autograd import Function
from typing import Dict, Any, Optional
import logging


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
# Configuration Management
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


# ============================================================================
# Scale Management
# ============================================================================

def save_scales(scales: Dict[str, float], scale_dir: str, layer_name: str):
    """
    Save quantization scales to file.
    
    Args:
        scales: Dictionary of scales (e.g., {'w_scale': 0.1, 'a_scale': 0.2})
        scale_dir: Directory to save scales
        layer_name: Name of the layer
    """
    os.makedirs(scale_dir, exist_ok=True)
    
    for scale_type, scale_value in scales.items():
        filename = f"{layer_name}_{scale_type}.p"
        filepath = os.path.join(scale_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(scale_value, f)


def load_scales(scale_dir: str, layer_name: str, scale_types: list) -> Dict[str, float]:
    """
    Load quantization scales from file.
    
    Args:
        scale_dir: Directory containing scales
        layer_name: Name of the layer
        scale_types: List of scale types to load (e.g., ['w_scale', 'a_scale'])
        
    Returns:
        Dictionary of loaded scales
    """
    scales = {}
    
    for scale_type in scale_types:
        filename = f"{layer_name}_{scale_type}.p"
        filepath = os.path.join(scale_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scale file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            scales[scale_type] = pickle.load(f)
    
    return scales


# ============================================================================
# Logging Utilities
# ============================================================================
"""
def setup_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Model Utilities
# ============================================================================

def get_layer_name_mapping() -> Dict[str, str]:
    return {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "out_proj": "out_proj",
        "fc1": "fc1",
        "fc2": "fc2",
        "lm_head": "lm_head",
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def get_model_size(model: torch.nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
"""

# ============================================================================
# Quantization Constants
# ============================================================================

# Shift numbers for fixed-point arithmetic
LINEAR_SHIFT_NUM = 2 ** 32
MATMUL_SHIFT_NUM = 2 ** 20

# Default quantization parameters
DEFAULT_DIGIT_SIZE = 4
DEFAULT_PARALLELISM = 4

