# -*- coding: utf-8 -*-
"""
RoBERTa IMDB Classification with Quantization Support

This script loads a custom quantized RoBERTa model and performs inference
on the IMDB dataset with various statistical analysis capabilities.
"""

import sys
import os
import importlib.util
import importlib
import time
import math
import pickle
import statistics
from typing import Dict, Optional, Tuple

import yaml
import nlp
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Quant.Roberta_quant import wrap_modules_in_net

# ============================================================================
# Configuration Constants
# ============================================================================

NUM_EPOCHS = 25
NUM_LAYERS = 12
DEFAULT_SAMPLE_NUM = 25000
TEST_DATASET_SIZE = 25000

# Model paths
HOST_ADDR = "/home//zhyzhao//Desktop//gpt2_roberta//Roberta_IMDB//"
MODEL_NAME = "aychang/roberta-base-imdb"

# Layer names for statistics collection
ATTENTION_LAYER_NAMES = ["Query_a", "key_a", "value_a", "Q", "P_o", 
                         "attn_out_a", "FCl1_a", "FCl2_a"]
SCALE_LAYER_NAMES = ["Query_a", "Query_w", "Query_o", "key_a", "key_w", 
                     "key_o", "value_a", "value_w", "value_o", "Q", "K",
                     "attention_scores", "P_o", "V_o", "attn_o", 
                     "attn_out_a", "attn_out_w", "attn_out_o",
                     "FCl1_a", "FCl1_w", "FCl1_o", "FCl2_a", "FCl2_w", "FCl2_o"]


# ============================================================================
# Module Replacement Setup
# ============================================================================

def setup_custom_modeling_module():
    """Replace transformers' modeling_roberta module with custom quantized version."""
script_dir = os.path.dirname(os.path.abspath(__file__))
quant_dir = os.path.join(script_dir, 'Quant')
custom_modeling_path = os.path.join(quant_dir, 'modeling_roberta.py')

    # Add Quant directory to sys.path
if quant_dir not in sys.path:
    sys.path.insert(0, quant_dir)

    # Import transformers base module first
import transformers

    # Clean up existing modules if present
if 'transformers.models.roberta.modeling_roberta' in sys.modules:
    del sys.modules['transformers.models.roberta.modeling_roberta']
if 'transformers.models.roberta' in sys.modules:
    roberta_module = sys.modules['transformers.models.roberta']
    if hasattr(roberta_module, 'modeling_roberta'):
        delattr(roberta_module, 'modeling_roberta')

# Load custom modeling_roberta module
spec = importlib.util.spec_from_file_location(
    "transformers.models.roberta.modeling_roberta", 
    custom_modeling_path
)
custom_modeling = importlib.util.module_from_spec(spec)

    # Register custom module
sys.modules['transformers.models.roberta.modeling_roberta'] = custom_modeling

    # Execute module loading
spec.loader.exec_module(custom_modeling)

# Ensure transformers.models.roberta uses the new module
if 'transformers.models.roberta' in sys.modules:
    transformers.models.roberta.modeling_roberta = custom_modeling


# Initialize custom module
setup_custom_modeling_module()


# ============================================================================
# Model Statistics Collection Class
# ============================================================================

class ModelStat:
    """
    Statistics collection and analysis for model activations.
    
    Supports various gating strategies and quantization schemes for
    analyzing model behavior during inference.
    """
    
    def __init__(self, name: str, bin_resolution: str = "gating_least_4b",
                 gating_strategy: str = "None", lower_threshold: float = 0,
                 upper_threshold: float = 0.1, quantization: str = "None",
                 quant_scale_dir: Optional[str] = None):
        """Initialize statistics collector."""
        self.sample_name = name
        self.sample_min = 0
        self.sample_max = 0
        self.bin_resolution = bin_resolution
        self.host_addr = HOST_ADDR
        self.hist_addr = self.host_addr + "Figures//token_digit_sparsity_plot//token_hist//"
        self.quant_scale_dir = quant_scale_dir
        
        # Initialize bins for histogram
        self.sample_bins = torch.arange(-256, 255, 1, dtype=torch.float32).cpu()
        self.sample_bins = torch.cat((self.sample_bins, torch.Tensor([255.5])), 0)
        
        # Statistics tracking
        self.sample_num = 0
        self.sample_shape = None
        self.sparsity = 0
        self.sample_hist = torch.Tensor([0])
        self.gating_strategy = gating_strategy
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.extracted = None
        self.quantization_scheme = quantization
        self.scale_sample = np.ndarray(DEFAULT_SAMPLE_NUM)
        self.token_len = 0
        self.inference_time = 0
        
        # EDN (Effective Digit Number) tracking
        self.EDN_d1en = 0
        self.EDN_d2en = 0
        self.EDN_d3en = 0
        self.EDN_d4en = 0
        self.EDN_d5en = 0
        self.EDN_d6en = 0
        self.EDN_d7en = 0
        self.EDN_d8en = 0
        self.var_EDN = None

    def _calculate_threshold(self) -> float:
        """Calculate threshold based on gating strategy."""
        if self.gating_strategy == "None":
            if self.bin_resolution == "gating_least_6b":
                length = self.sample_max - self.sample_min
                return self.sample_min + length / 4
            elif self.bin_resolution == "gating_least_4b":
                length = self.sample_max - self.sample_min
                return self.sample_min + length / 16
        elif self.gating_strategy == "threshold_gating":
            return self.upper_threshold
        return 0.0
    
    def sample_range_check(self, sample: torch.Tensor):
        """Check and update sample range."""
        if self.gating_strategy == "None":
            self.sample_min = min(self.sample_min, torch.min(sample))
            self.sample_max = max(self.sample_max, torch.max(sample))
            self.token_len = self.token_len + sample.shape[1]
            self.sample_num = self.sample_num + 1

    def scale_check(self, sample: torch.Tensor):
        """Check scale of sample."""
        self.scale_sample[self.sample_num] = sample.cpu().detach().numpy()
        self.sample_num = self.sample_num + 1

    def sparsity_check(self, sample: torch.Tensor):
        """Calculate sparsity of sample."""
        threshold = self._calculate_threshold()
        sparsity_mat = torch.where(sample[0] < threshold, 0, 1)
        sample_sparsity = sparsity_mat.sum().item() / sparsity_mat.numel()
        self.sparsity = self.sparsity + sample_sparsity
        self.sample_num = self.sample_num + 1

    def row_sparsity_check(self, sample: torch.Tensor):
        """Calculate row-wise sparsity."""
        threshold = self._calculate_threshold()
        sparsity_mat = torch.where(sample < threshold, 0, 1)

        row_sparsity = []
        for batch in sparsity_mat:
            for head in batch:
                for rows in head:
                    row_sparsity.append(rows.sum().item())

        average_row_sparsity = statistics.mean(row_sparsity)
        self.sparsity = self.sparsity + average_row_sparsity
        self.sample_num = self.sample_num + 1

    def bin_gating(self, sample: torch.Tensor, bins: torch.Tensor, gating_bin: int = 0) -> torch.Tensor:
        """Apply bin-based gating to sample."""
        sample = torch.where(
            torch.logical_and(sample > bins[gating_bin - 1].item(), 
                            sample < bins[gating_bin].item()),
            bins[gating_bin].cuda(), sample
        )
        sample = torch.where(
            torch.logical_and(sample > bins[gating_bin].item(), 
                            sample < bins[gating_bin + 1].item()),
            bins[gating_bin + 1].cuda(), sample
        )
        return sample

    def threshold_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply threshold-based gating."""
        sample = torch.where(
            torch.logical_and(sample > self.lower_threshold, 
                            sample < self.upper_threshold),
            torch.zeros(1, dtype=torch.float32).cuda(), sample
        )
        return sample

    def P_token_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply token-level gating based on max values."""
        gating_k = torch.zeros(1, dtype=torch.float32).cuda()
        gating_k[0] = self.upper_threshold
        token_max = torch.max(sample[0], dim=2)
        token_gating_threshold = gating_k.repeat((sample.shape[2], 12)).T
        token_invalid = torch.le(token_max.values, token_gating_threshold)
        token_gating_idx = torch.nonzero(token_invalid)
        sample[0, token_gating_idx[:, 0], token_gating_idx[:, 1], :] = \
            torch.zeros(1, dtype=torch.float32).cuda()
        token_sparsity = token_gating_idx.shape[0] / (sample.shape[2] * sample.shape[1])
        self.sparsity = self.sparsity + token_sparsity
        return sample

    def P_token_max_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply token-level gating using kth value."""
        gating_k = math.floor(sample.shape[2] * self.upper_threshold)
        token_max = torch.max(sample[0], dim=2)
        token_gating_threshold = torch.kthvalue(token_max.values, k=gating_k, dim=1).values
        token_gating_threshold = token_gating_threshold.repeat((sample.shape[2], 1)).T
        token_invalid = torch.le(token_max.values, token_gating_threshold)
        token_gating_idx = torch.nonzero(token_invalid)
        sample[0, token_gating_idx[:, 0], token_gating_idx[:, 1], :] = \
            torch.zeros(1, dtype=torch.float32).cuda()
        return sample

    def P_token_var_max_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply token-level gating based on variance."""
        gating_k = math.floor(sample.shape[2] * self.upper_threshold)
        token_var = -torch.var(sample[0], dim=-1)
        token_gating_threshold = torch.kthvalue(token_var, k=gating_k, dim=1).values
        token_gating_threshold = token_gating_threshold.repeat((sample.shape[2], 1)).T
        token_invalid = torch.le(token_var, token_gating_threshold)
        token_gating_idx = torch.nonzero(token_invalid)
        sample[0, token_gating_idx[:, 0], token_gating_idx[:, 1], :] = \
            torch.zeros(1, dtype=torch.float32).cuda()
        return sample

    def qkv_feature_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply feature-level gating for QKV."""
        gating_k = math.floor(sample[0].shape[2] * self.upper_threshold)
        column_mean = torch.abs(torch.mean(sample[0][0], dim=0))
        feature_gating_threshold = torch.kthvalue(column_mean, k=gating_k, dim=0).values
        feature_invalid = column_mean < feature_gating_threshold
        feature_gating_idx = torch.nonzero(feature_invalid)
        sample[0][0, :, feature_gating_idx] = torch.zeros(1, dtype=torch.float32).cuda()
        return sample

    def qkv_token_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply token-level gating for QKV."""
        gating_k = math.floor(sample[0].shape[1] * self.upper_threshold)
        column_varmean = torch.mean(sample[0][0], dim=1)
        feature_gating_threshold = torch.kthvalue(column_varmean, k=gating_k, dim=0).values
        token_invalid = column_varmean < feature_gating_threshold
        token_gating_idx = torch.nonzero(token_invalid)
        sample[0][0, token_gating_idx, :] = torch.zeros(1, dtype=torch.float32).cuda()
        return sample

    def P_feature_gating(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply feature-level gating."""
        gating_k = math.floor(sample.shape[2] * self.upper_threshold)
        column_mean = torch.mean(sample[0], dim=1)
        feature_gating_threshold = torch.kthvalue(column_mean, k=gating_k, dim=1).values
        feature_gating_threshold = feature_gating_threshold.repeat((sample.shape[2], 1)).T
        feature_invalid = torch.le(column_mean, feature_gating_threshold)
        feature_gating_idx = torch.nonzero(feature_invalid)
        sample[0, feature_gating_idx[:, 0], :, feature_gating_idx[:, 1]] = \
            torch.zeros(1, dtype=torch.float32).cuda()
        return sample

    def sample_distribution(self, sample: torch.Tensor):
        """Calculate sample distribution histogram."""
        sample = sample.cpu()
        hist = torch.histogram(sample, bins=self.sample_bins)
        self.sample_hist = self.sample_hist + hist.hist / 10000

    def token_distribution(self, sample: torch.Tensor):
        """Calculate token distribution and inference time."""
        sample = sample.cpu()
        hist = torch.histogram(sample, bins=self.sample_bins)
        EBW_list = np.zeros(8)
        
        # Calculate effective bit width distribution
        EBW_list[0] = hist.hist[256 - 2:256 + 2].sum()
        EBW_list[1] = (hist.hist[256 - 4:256 - 2].sum() + hist.hist[256 + 2:256 + 4].sum())
        EBW_list[2] = (hist.hist[256 - 8:256 - 4].sum() + hist.hist[256 + 4:256 + 8].sum())
        EBW_list[3] = (hist.hist[256 - 16:256 - 8].sum() + hist.hist[256 + 8:256 + 16].sum())
        EBW_list[4] = (hist.hist[256 - 32:256 - 16].sum() + hist.hist[256 + 16:256 + 32].sum())
        EBW_list[5] = (hist.hist[256 - 64:256 - 32].sum() + hist.hist[256 + 32:256 + 64].sum())
        EBW_list[6] = (hist.hist[256 - 128:256 - 64].sum() + hist.hist[256 + 64:256 + 128].sum())
        EBW_list[7] = (hist.hist[256 - 256:256 - 128].sum() + hist.hist[256 + 128:256 + 255].sum())
        
        self.inference_time = self.inference_time + sum(EBW_list) * 8
    
    def EDN_analysis(self, sample: torch.Tensor):
        """Effective Digit Number (EDN) analysis for 12-bit, digit_size=4."""
        # Calculate EDN for different digit positions
        token_EDN_d1en = (~torch.bitwise_and((sample < 256), (sample >= -256)))
        token_EDN_d2en = (~((sample.abs() % 256) < 16))
        token_EDN_d3en = (~((sample.abs() % 16) == 0))

        k = 4
        p1d = (0, 0, 0, k - token_EDN_d1en.shape[-2] % k)
        token_EDN_d1en = torch.nn.functional.pad(token_EDN_d1en, p1d, mode="constant", value=0)
        token_EDN_d2en = torch.nn.functional.pad(token_EDN_d2en, p1d, mode="constant", value=0)
        token_EDN_d3en = torch.nn.functional.pad(token_EDN_d3en, p1d, mode="constant", value=0)
        
        # Calculate parallel sparsity for each digit
        for i, token_EDN in enumerate([token_EDN_d1en, token_EDN_d2en, token_EDN_d3en], 1):
            parallel_token_spar = torch.stack(list(torch.split(token_EDN, k, dim=-2)))
            parallel_token_spar = parallel_token_spar.sum(dim=-2)
            Np = parallel_token_spar.flatten().shape[0]
            parallel_token_spar = (parallel_token_spar == 0)
            parallel_token_spar = parallel_token_spar.sum() / Np
            
            setattr(self, f'EDN_d{i}en', getattr(self, f'EDN_d{i}en') + parallel_token_spar)
    
    def var_EDN_analysis(self, sample: torch.Tensor) -> torch.Tensor:
        """Variance-based EDN analysis."""
        token_var = torch.var(sample, dim=-1)
        token_EDN_shape = token_var.shape

        token_EDN_d1en = (~(sample.abs() < 64)).sum(dim=-1)
        token_EDN_d2en = (~((sample % 64) < 8)).sum(dim=-1)
        token_EDN_d3en = (~((sample % 8) == 0)).sum(dim=-1)

        EDN = token_EDN_d1en + token_EDN_d2en + token_EDN_d3en
        var_EDN_sample = torch.stack((token_var, EDN), dim=0)
        
        if self.var_EDN is None:
            self.var_EDN = var_EDN_sample
        else:
            self.var_EDN = torch.cat((self.var_EDN, var_EDN_sample), dim=-1)

        return sample
    
    def quantization(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply quantization to sample."""
        return torch.floor(sample)
    
    def statistic_probe(self, sample: torch.Tensor) -> torch.Tensor:
        """Main probe function that routes to appropriate analysis method."""
        # Route to appropriate gating/analysis strategy
        strategy_map = {
            "None": lambda s: (self.sparsity_check(s), s)[1],
            "scale_inspect": lambda s: (self.scale_check(s), s)[1],
            "bin_gating": lambda s: self.bin_gating(s, self.sample_bins, gating_bin=15),
            "threshold_gating": lambda s: self.threshold_gating(s),
            "P_token_gating": lambda s: self.P_token_gating(s),
            "P_token_var_max_gating": lambda s: self.P_token_var_max_gating(s),
            "qkv_feature_gating": lambda s: self.qkv_feature_gating(s),
            "P_feature_gating": lambda s: self.P_feature_gating(s),
            "P_token_max_gating": lambda s: self.P_token_max_gating(s),
            "quantization": lambda s: (self.sample_distribution(s), s)[1],
            "token_hist": lambda s: (self.token_distribution(s), s)[1],
            "EDN_analysis": lambda s: (self.EDN_analysis(s), s)[1],
            "var_EDN": lambda s: self.var_EDN_analysis(s),
        }
        
        if self.gating_strategy in strategy_map:
            return strategy_map[self.gating_strategy](sample)
        
        return sample

    def put_overall_sparsity(self):
        """Print and log overall sparsity statistics."""
        if self.gating_strategy in ["P_token_gating", "qkv_feature_gating"]:
            sparsity = self.sparsity / DEFAULT_SAMPLE_NUM
            msg = f"The overall token sparsity of //{self.sample_name}// on threshold {self.upper_threshold} is {sparsity}"
        else:
            try:
                sparsity = self.sparsity / self.sample_num
            except ZeroDivisionError:
                sparsity = 0
            msg = f"The overall sparsity of //{self.sample_name}// on threshold {self.upper_threshold} is {1 - sparsity}"
        
        print(msg)
        with open("log.txt", "a") as f:
            f.write(msg + '\n')

    def put_overall_EDN(self):
        """Save EDN analysis results."""
        if self.gating_strategy == "EDN_analysis":
            sen_emb_test_filename = (
                self.host_addr + "digit_sparsity//12a8wd4//12a8wd4k4//" + 
                self.sample_name + ".p"
            )
            EDN = [
                self.EDN_d1en / self.sample_num,
                self.EDN_d2en / self.sample_num,
                self.EDN_d3en / self.sample_num
            ]
            print(f"EDN of {self.sample_name} is {EDN}")
            pickle.dump(EDN, open(sen_emb_test_filename, "wb"))

    def put_var_EDN(self):
        """Save variance-based EDN results."""
        if self.gating_strategy == "var_EDN":
            sen_emb_test_filename = (
                self.host_addr + "Roberta_IMDB//Roberta_output//var_EDN//" + 
                self.sample_name + ".p"
            )
            pickle.dump(self.var_EDN, open(sen_emb_test_filename, "wb"))

    def put_overall_scale(self):
        """Save scale inspection results."""
        if self.gating_strategy == "scale_inspect":
            if self.quant_scale_dir is None:
                raise ValueError("quant_scale_dir must be set for scale inspection")
            sen_emb_test_filename = (
                self.quant_scale_dir + 
                self.sample_name + ".p"
            )
            pickle.dump(self.scale_sample.max(), open(sen_emb_test_filename, "wb"))

    def put_overall_distribution(self):
        """Save distribution histogram."""
        if self.gating_strategy == "None":
            sen_emb_test_filename = (
                self.host_addr + "Roberta_IMDB//Roberta_output//sample_distribution//" + 
                self.sample_name + ".p"
            )
            pickle.dump(self.sample_hist.cpu(), open(sen_emb_test_filename, "wb"))

    def put_token_distribution(self) -> float:
        """Print and return token distribution inference time."""
        if self.gating_strategy == "token_hist":
            print(f"inference time on {self.sample_name} is {self.inference_time}")
        return self.inference_time


# ============================================================================
# Dataset and Model Loading
# ============================================================================

class IMDBDataset:
    """IMDB movie review dataset wrapper."""
    
    def __init__(self, part: str):
        """Initialize dataset for train/test split."""
        self.dataset = nlp.load_dataset('imdb')[part]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        review = self.dataset[idx]
        label = torch.tensor(review['label'])
        text = torch.tensor(
            self.tokenizer.encode(review['text'], padding=True, truncation=True)
        )
        return {'text': text, 'label': label}

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset.num_rows

    def plot_seq_length(self):
        """Plot sequence length distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist([len(sample) for sample in list(self.dataset['text'])], 100)
        plt.xlabel('Length of samples')
        plt.ylabel('Number of samples')
        plt.title('Sample length distribution')
        plt.show()


def load_roberta_model():
    """Load and return RoBERTa model for IMDB classification."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.cuda()
    return model


def get_dataloader(part: str, batch_size: int = 1) -> Tuple[DataLoader, IMDBDataset]:
    """Create and return dataloader for specified dataset part."""
    dataset = IMDBDataset(part)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset


# ============================================================================
# Statistics Initialization
# ============================================================================

def create_edn_stat_list(gating_strategy: str = "None") -> Dict[str, ModelStat]:
    """Create statistics list for EDN analysis."""
    stat_list = {}
    
    # Create stats for each layer
    for layer_idx in range(NUM_LAYERS):
        for layer_name in ATTENTION_LAYER_NAMES:
            key = f"{layer_name}_{layer_idx}"
            stat_list[key] = ModelStat(name=key, gating_strategy=gating_strategy)
    
    # Add output projection layer
    stat_list["output_proj1_a_0"] = ModelStat(
        name="output_proj1_a_0", gating_strategy=gating_strategy
    )
    
    return stat_list


def create_scale_stat_list(quant_scale_dir: Optional[str] = None) -> Dict[str, ModelStat]:
    """Create statistics list for scale inspection."""
    stat_list = {}
    
    # Create stats for each layer
    for layer_idx in range(NUM_LAYERS):
        for layer_name in SCALE_LAYER_NAMES:
            key = f"{layer_name}_scale_{layer_idx}"
            stat_list[key] = ModelStat(
                name=key, 
                gating_strategy="scale_inspect",
                quant_scale_dir=quant_scale_dir
            )
    
    # Add output projection layers
    for suffix in ["a", "w", "o"]:
        key = f"output_proj1_{suffix}_scale_0"
        stat_list[key] = ModelStat(
            name=key, 
            gating_strategy="scale_inspect",
            quant_scale_dir=quant_scale_dir
        )
    
    return stat_list


def initialize_stat_list(
    threshold: float, 
    scale_inspect: bool = False,
    quant_scale_dir: Optional[str] = None,
    gating_strategy: str = "None"
) -> Dict[str, ModelStat]:
    """Initialize statistics collection list."""
    if scale_inspect:
        return create_scale_stat_list(quant_scale_dir=quant_scale_dir)
    else:
        return create_edn_stat_list(gating_strategy=gating_strategy)


def stat_list_conclusion(stat_list: Dict[str, ModelStat], dump_idx: int = 0, calibration: bool = False):
    """Conclude and save statistics from all collectors."""
    for stat in stat_list.values():
        #stat.put_overall_EDN()
        if calibration:
            stat.put_overall_scale()
        else:
            #stat.put_overall_scale()
            pass


# ============================================================================
# Testing Functions
# ============================================================================

def test_epoch(
    model, 
    dataloader, 
    dataset, 
    dumpfile_name: str, 
    threshold: float, 
    enable_quantization: bool = False, 
    calibration: bool = False,
    quant_scale_dir: Optional[str] = None,
    gating_strategy: str = "None"
) -> float:
    """Run one test epoch and return accuracy."""
    corr = 0
    stat_list = initialize_stat_list(
        threshold, 
        scale_inspect=calibration,
        quant_scale_dir=quant_scale_dir,
        gating_strategy=gating_strategy
    )
    start = time.time()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            tokens, labels = batch['text'], batch['label']
            tokens = tokens.cuda()
            labels = labels.cuda()

            model.zero_grad()
            output = model(tokens, model_stat=stat_list)
            predictions = torch.argmax(output.logits, dim=-1)
            
            if predictions == labels:
                corr += 1

            if idx % 10 == 0:
                print(f"batch {idx} finished")

    if enable_quantization:
        stat_list_conclusion(stat_list, 0, calibration)
    else:
        print(f"Floating point model inference finished.")

    accuracy = corr / TEST_DATASET_SIZE
    end = time.time()
    
    print(f"The accuracy is {accuracy}")
    print(f"Roberta inference finished, elapsed time = {end - start}")
    return accuracy


def test(
    enable_quantization: bool = False, 
    quant_scale_dir: Optional[str] = None, 
    calibration: bool = False, 
    gating_strategy: str = "None",
    quant_bit_config: Optional[Dict[str, Dict[str, int]]] = None,
    config_type: str = "final"
):
    """
    Main test function.
    
    Args:
        enable_quantization: If True, apply quantization to the model. 
                            Default is False (no quantization).
        quant_scale_dir: Directory path for quantization scale files.
                        Required if enable_quantization is True.
        calibration: If True, run in calibration mode to collect quantization scales.
        gating_strategy: Strategy for feature gating.
        quant_bit_config: Dictionary of bit width configurations for each module type.
                         If None, uses default configurations from Roberta_quant.py.
    """

    # Load model
    model = load_roberta_model()
    
    # Wrap model with quantization (only if enabled)
    if enable_quantization:
        if quant_scale_dir is None:
            raise ValueError("quant_scale_dir must be provided when enable_quantization is True")
        if(calibration):    
            quant_type="scale_insp"
        else:
            quant_type="qlinear"
        wrapped_model = wrap_modules_in_net(
            model, 
            linear_layer_quant=quant_type, 
            enable_quantization=True,
            quant_scale_dir=quant_scale_dir,
            quant_bit_config=quant_bit_config
        )
        print(f"Model wrapped with quantization enabled. Scale directory: {quant_scale_dir} \n")
    else:
        print("Using original model without quantization. \n")
    
    # Print model information to console and file
    print(model)
    with open("model_information.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Model Information\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Quantization Enabled: {enable_quantization}\n")
        if enable_quantization:
            f.write(f"Quantization Scale Directory: {quant_scale_dir}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Architecture:\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Model Parameters Summary:\n")
        f.write("=" * 80 + "\n\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    # Get dataloader
    dataloader, dataset = get_dataloader('test')
    
    # Test with different thresholds
    thre_list = [0]  # Can be expanded: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    test_acc = []
    
    with open("./log.txt", "a") as f:
        for i, threshold in enumerate(thre_list):
            print(f"feature_gating test {i} start!")
            dumpfilename = f"feature_gating_{i}"
            acc = test_epoch(
                model, 
                dataloader, 
                dataset, 
                dumpfilename, 
                threshold, 
                enable_quantization,
                calibration,
                quant_scale_dir=quant_scale_dir,
                gating_strategy=gating_strategy
            )
            test_acc.append(acc)
            f.write(f"The test accuracy under threshold {thre_list[i]} is {acc} ")
            print(f"P_var_max_gating test {i} finished!\n")
    
    print(f"The overall test accuracy is {test_acc} at config {config_type}")
    with open("./log.txt", "a") as f:
        f.write(f"thre_list = {thre_list} ")
        f.write(f"The overall test accuracy is {test_acc} at config {config_type}, calibration: {calibration}  \n")
        if(calibration):  
            f.write(f"Calibrate the quantized model, the scale is saved in {quant_scale_dir} \n")
        else:
            f.write(f"Inferenced with quantized model \n")

# ============================================================================
# Configuration Loading
# ============================================================================

def load_quant_config(config_path: str = "config/final.yaml") -> Dict:
    """
    Load quantization configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file (relative to script directory)
        
    Returns:
        Dictionary containing quantization configuration parameters
    """
    # Get the absolute path relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_config_path = os.path.join(script_dir, config_path)
    
    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"Configuration file not found: {abs_config_path}")
    
    with open(abs_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configuration
    # Set enable_quantization=True to enable model quantization
    # Quantization scale directory (required if enable_quantization=True)
    # Update this path to point to your quantization scale files directory
    # Initialize log file
    with open("log.txt", "w") as f:
        f.write("log start\n\n")
    
    
    config_type = "Int8"
    QUANT_SCALE_DIR = "//home//zyzhao//Desktop//HPCA2026_FOCUS//Quant//quant_scale//" + config_type + "//"
    Config_path = "./config/"+ config_type +".yaml"
    # Load quantization bit width configuration from YAML file
    # Configuration file: config/final.yaml
    try:
        quant_config = load_quant_config(Config_path)
        Q_quant_bit = quant_config["Q_quant_bit"]
        K_quant_bit = quant_config["K_quant_bit"]
        V_quant_bit = quant_config["V_quant_bit"]
        AOD_quant_bit = quant_config["AOD_quant_bit"]
        ID_quant_bit = quant_config["ID_quant_bit"]
        OD_quant_bit = quant_config["OD_quant_bit"]
        CD_quant_bit = quant_config["CD_quant_bit"]
        QK_quant_bit = quant_config["QK_quant_bit"]
        PV_quant_bit = quant_config["PV_quant_bit"]
        Digit_size = quant_config["Digit_size"]
        Parallelism = quant_config["Parallelism"]
        print(f"Successfully loaded quantization configuration from {Config_path}")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default values...")
        # Fallback to default values
        Q_quant_bit = [8, 4, 6]
        K_quant_bit = [8, 4, 8]
        V_quant_bit = [8, 8, 8]
        AOD_quant_bit = [6, 4, 8]
        ID_quant_bit = [8, 8, 8]
        OD_quant_bit = [8, 8, 8]
        CD_quant_bit = [8, 8, 8]
        QK_quant_bit = [6, 8, 10]
        PV_quant_bit = [10, 8, 6]
        Digit_size = 4
        Parallelism = 4
    
    QUANT_BIT_CONFIG = {
        # Scale inspection modules
        "scale_insp_q": {"a_bit": Q_quant_bit[0], "w_bit": Q_quant_bit[1], "o_bit": Q_quant_bit[2]},
        "scale_insp_k": {"a_bit": K_quant_bit[0], "w_bit": K_quant_bit[1], "o_bit": K_quant_bit[2]},
        "scale_insp_v": {"a_bit": V_quant_bit[0], "w_bit": V_quant_bit[1], "o_bit": V_quant_bit[2]},
        "scale_insp_aod": {"a_bit": AOD_quant_bit[0], "w_bit": AOD_quant_bit[1], "o_bit": AOD_quant_bit[2]},
        "scale_insp_id": {"a_bit": ID_quant_bit[0], "w_bit": ID_quant_bit[1], "o_bit": ID_quant_bit[2]},
        "scale_insp_od": {"a_bit": OD_quant_bit[0], "w_bit": OD_quant_bit[1], "o_bit": OD_quant_bit[2]},
        "scale_insp_cd": {"a_bit": CD_quant_bit[0], "w_bit": CD_quant_bit[1], "o_bit": CD_quant_bit[2]},
        "scale_insp_qk": {"A_bit": QK_quant_bit[0], "B_bit": QK_quant_bit[1], "O_bit": QK_quant_bit[2]},
        "scale_insp_PV": {"A_bit": PV_quant_bit[0], "B_bit": PV_quant_bit[1], "O_bit": PV_quant_bit[2]},
        # Quantized forward modules
        "qlinear_q": {"a_bit": Q_quant_bit[0], "w_bit": Q_quant_bit[1], "o_bit": Q_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_k": {"a_bit": K_quant_bit[0], "w_bit": K_quant_bit[1], "o_bit": K_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_v": {"a_bit": V_quant_bit[0], "w_bit": V_quant_bit[1], "o_bit": V_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_attn_out": {"a_bit": AOD_quant_bit[0], "w_bit": AOD_quant_bit[1], "o_bit": AOD_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_FC1": {"a_bit": Q_quant_bit[0], "w_bit": Q_quant_bit[1], "o_bit": Q_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_FC2": {"a_bit": Q_quant_bit[0], "w_bit": Q_quant_bit[1], "o_bit": Q_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_LMhead": {"a_bit": Q_quant_bit[0], "w_bit": Q_quant_bit[1], "o_bit": Q_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_QKmul": {"A_bit": QK_quant_bit[0], "B_bit": QK_quant_bit[1], "O_bit": QK_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
        "qlinear_PVmul": {"A_bit": PV_quant_bit[0], "B_bit": PV_quant_bit[1], "O_bit": PV_quant_bit[2], "d_bit": Digit_size, "p": Parallelism},
    }
    
    # Run test
    test(
        enable_quantization=True, 
        quant_scale_dir=QUANT_SCALE_DIR, 
        calibration=True, 
        gating_strategy="None",
        quant_bit_config=QUANT_BIT_CONFIG,
        config_type=config_type
    )

    test(
        enable_quantization=True, 
        quant_scale_dir=QUANT_SCALE_DIR, 
        calibration=False, 
        gating_strategy="None",
        quant_bit_config=QUANT_BIT_CONFIG,
        config_type=config_type
    )
 
    print(f"inference finished!")