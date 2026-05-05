"""
Quantized Matrix Multiplication Implementation.

This module provides quantized matrix multiplication layers for:
- Attention score computation: Q @ K^T
- Attention output computation: P @ V

Supported modes:
- raw
- scale_inspection
- quant_forward

This version uses torch.matmul instead of torch.bmm, so it supports:
- 2D: [M, K] @ [K, N]
- 3D: [B, M, K] @ [B, K, N]
- 4D: [B, H, M, K] @ [B, H, K, N]
"""

import json
import os
import pickle
from typing import Optional

import torch
import torch.nn as nn

from .utils import Round, MATMUL_SHIFT_NUM


class MatMul(nn.Module):
    """Simple non-quantized matrix multiplication module."""

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


class QuantizedMatMul(nn.Module):
    """
    Quantized matrix multiplication layer.

    This layer is intended for:
      - qk_matmul: Q @ K^T
      - pv_matmul: P @ V

    It uses symmetric per-tensor quantization for A, B, and output O.
    """

    def __init__(
        self,
        mode: str = "raw",
        A_bit: Optional[int] = None,
        B_bit: Optional[int] = None,
        O_bit: Optional[int] = None,
        scale_root_str: Optional[str] = None,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        outlier_ratio: float = 0.0,
    ):
        super().__init__()

        self.mode = mode

        self.A_bit = A_bit
        self.B_bit = B_bit
        self.O_bit = O_bit

        self.digit_size = d_bit
        self.parallelism = p

        self.scale_root_str = scale_root_str or ""
        self.round = Round

        self.A_interval: Optional[float] = None
        self.B_interval: Optional[float] = None
        self.O_interval: Optional[float] = None

        if self.A_bit is not None:
            self.A_qmax = 2 ** (self.A_bit - 1)
        else:
            self.A_qmax = None

        if self.B_bit is not None:
            self.B_qmax = 2 ** (self.B_bit - 1)
        else:
            self.B_qmax = None

        if self.O_bit is not None:
            self.O_qmax = 2 ** (self.O_bit - 1)
        else:
            self.O_qmax = None

        self.layer_name = ""
        self.layer_idx = 0

        # 第一版 MatMul 量化先不要启用 outlier 分支。
        self.outlier_ratio = outlier_ratio

        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None

    def set_layer_info(self, layer_name: str, layer_idx: int):
        self.layer_name = layer_name
        self.layer_idx = layer_idx

    def _matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        General batched matmul.

        Supports 2D / 3D / 4D / higher-dimensional batched matmul.
        """
        return torch.matmul(A, B)

    def _check_bits(self):
        if self.A_bit is None or self.B_bit is None or self.O_bit is None:
            raise ValueError(
                f"{self.layer_name}_{self.layer_idx}: "
                f"A_bit/B_bit/O_bit must be set. "
                f"Got A_bit={self.A_bit}, B_bit={self.B_bit}, O_bit={self.O_bit}."
            )

    @staticmethod
    def _safe_interval_from_tensor(x: torch.Tensor, qmax: int) -> float:
        max_abs = x.detach().abs().max()
        if max_abs == 0 or torch.isnan(max_abs) or torch.isinf(max_abs):
            return 1.0
        return (max_abs / (qmax - 0.5)).item()

    def _scale_file_paths(self):
        return (
            os.path.join(
                self.scale_root_str,
                f"{self.layer_name}_A_scale_{self.layer_idx}.p",
            ),
            os.path.join(
                self.scale_root_str,
                f"{self.layer_name}_B_scale_{self.layer_idx}.p",
            ),
            os.path.join(
                self.scale_root_str,
                f"{self.layer_name}_O_scale_{self.layer_idx}.p",
            ),
        )

    def _scale_files_exist(self) -> bool:
        return all(os.path.exists(p) for p in self._scale_file_paths())

    def _resolve_calibration_action(self) -> str:
        if self._calibration_action_cache is not None:
            return self._calibration_action_cache

        policy = str(self.calibration_policy).lower()

        if policy == "auto":
            action = "reuse" if self._scale_files_exist() else "recalibrate"
        elif policy in {"reuse", "recalibrate"}:
            action = policy
        else:
            raise ValueError(
                f"Invalid calibration_policy={self.calibration_policy} "
                f"for {self.layer_name}_{self.layer_idx}"
            )

        self._calibration_action_cache = action
        return action

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        collect_stats: bool = False,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        if stat_collector is None and hasattr(self, "_stat_manager"):
            stat_collector = self._stat_manager

        if self.mode == "raw":
            return self._matmul(A, B)

        if self.mode == "scale_inspection":
            return self.scale_inspection(A, B, stat_collector)

        if self.mode == "quant_forward":
            return self.quant_forward(A, B, stat_collector)

        raise NotImplementedError(f"Mode {self.mode} not implemented")

    def quant_input(
        self,
        x: torch.Tensor,
        interval: float,
        qmax: int,
    ) -> torch.Tensor:
        return (x / interval).round().clamp(-qmax, qmax - 1)

    def scale_inspection(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Collect A/B/O scales.

        A_interval is collected from A.
        B_interval is collected from B.
        O_interval is collected from raw output O = A @ B.
        """
        self._check_bits()

        action = self._resolve_calibration_action()
        if action == "reuse":
            return self._matmul(A, B)

        if self.outlier_ratio > 0.0:
            raise NotImplementedError(
                "QuantizedMatMul outlier path is disabled in the stable baseline. "
                "Set qk_matmul/pv_matmul outlier_ratio=0.0."
            )

        out = self._matmul(A, B)

        self.A_interval = self._safe_interval_from_tensor(A, self.A_qmax)
        self.B_interval = self._safe_interval_from_tensor(B, self.B_qmax)
        self.O_interval = self._safe_interval_from_tensor(out, self.O_qmax)

        if stat_collector is not None:
            stat_collector.collect_matmul_stats(
                self.layer_name,
                self.layer_idx,
                self.A_interval,
                self.B_interval,
                self.O_interval,
            )

        return out

    def quant_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Quantized matmul.

        Real approximation:
            A ~= A_interval * A_q
            B ~= B_interval * B_q
            O ~= O_interval * O_q

        Integer-style path:
            acc = A_q @ B_q
            O_q = round_or_floor(acc * A_interval * B_interval / O_interval)
            O = O_q * O_interval
        """
        self._check_bits()
        self._load_scales()

        if self.outlier_ratio > 0.0:
            raise NotImplementedError(
                "QuantizedMatMul outlier path is disabled in the stable baseline. "
                "Set qk_matmul/pv_matmul outlier_ratio=0.0."
            )

        M0 = torch.tensor(
            self.A_interval * self.B_interval / self.O_interval,
            device=A.device,
            dtype=torch.float32,
        )
        M0 = self.round(M0 * MATMUL_SHIFT_NUM)

        A_sim = self.quant_input(A, self.A_interval, self.A_qmax).to(torch.float32)
        B_sim = self.quant_input(B, self.B_interval, self.B_qmax).to(torch.float32)

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                A_sim,
                self.A_bit,
                self.digit_size,
                self.parallelism,
            )
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                B_sim,
                self.B_bit,
                self.digit_size,
                self.parallelism,
            )

        out_quant = self._matmul(A_sim, B_sim)
        out_quant = out_quant.mul(M0)
        out_quant = torch.div(
            out_quant,
            MATMUL_SHIFT_NUM,
            rounding_mode="floor",
        )

        out = out_quant.mul(self.O_interval).to(A.dtype)

        with torch.no_grad():
            out_ref = self._matmul(A, B)
            mse = ((out - out_ref) ** 2).mean().item()
            self.log_quant_error(
                mse,
                layer_name=self.layer_name,
                layer_idx=self.layer_idx,
            )

        return out

    def _load_scales(self):
        A_scale_file, B_scale_file, O_scale_file = self._scale_file_paths()

        missing = [
            path for path in (A_scale_file, B_scale_file, O_scale_file)
            if not os.path.exists(path)
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing scale files for {self.layer_name}_{self.layer_idx}: "
                f"{missing}"
            )

        with open(A_scale_file, "rb") as f:
            self.A_interval = pickle.load(f)

        with open(B_scale_file, "rb") as f:
            self.B_interval = pickle.load(f)

        with open(O_scale_file, "rb") as f:
            self.O_interval = pickle.load(f)

        if self.A_interval is None or self.B_interval is None or self.O_interval is None:
            raise ValueError(
                f"Invalid loaded scales for {self.layer_name}_{self.layer_idx}: "
                f"A={self.A_interval}, B={self.B_interval}, O={self.O_interval}"
            )

    def save_scales(self):
        if self.A_interval is None or self.B_interval is None or self.O_interval is None:
            raise ValueError(
                f"Scales not computed for {self.layer_name}_{self.layer_idx}. "
                f"Run scale_inspection first."
            )

        os.makedirs(self.scale_root_str, exist_ok=True)

        A_scale_file, B_scale_file, O_scale_file = self._scale_file_paths()

        with open(A_scale_file, "wb") as f:
            pickle.dump(self.A_interval, f)

        with open(B_scale_file, "wb") as f:
            pickle.dump(self.B_interval, f)

        with open(O_scale_file, "wb") as f:
            pickle.dump(self.O_interval, f)

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, "
            f"A_bit={self.A_bit}, "
            f"B_bit={self.B_bit}, "
            f"O_bit={self.O_bit}, "
            f"layer={self.layer_name}_{self.layer_idx}"
        )

    def log_quant_error(
        self,
        value: float,
        layer_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        """
        Append quantization MSE log.

        This keeps the same lightweight behavior as your existing implementation.
        """
        layer_name = layer_name if layer_name is not None else self.layer_name
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx

        if self.scale_root_str is None or self.scale_root_str == "":
            return

        try:
            os.makedirs(self.scale_root_str, exist_ok=True)
            log_file = os.path.join(self.scale_root_str, "quant_error_log.jsonl")

            record = {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "mse": float(value),
                "mode": self.mode,
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception:
            # Logging should never break inference.
            pass