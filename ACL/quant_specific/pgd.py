import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from transformers import TrainerCallback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


@dataclass
class QuantizeArguments:
    attack_step: Optional[Literal["injection", "removal"]] = field(default=None)
    attack_strategy: Optional[Literal["default", "unlearn", "ghost"]] = field(default="default")


def compute_pruning_mask(model, pruning_ratio, target_param_suffixes=None):
    """
    全局幅度剪枝掩码计算。
    - 收集所有 model.layers.*.weight 的绝对值
    - 计算阈值 τ = kthvalue(all_magnitudes, k=P*N)
    - 返回 mask_dict: {name: BoolTensor}，True=M=1（大/保留），False=M=0（小/幽灵）
    - 返回 threshold: float τ
    """
    if target_param_suffixes is None:
        target_param_suffixes = [".weight"]

    all_magnitudes = []
    target_names = []
    for name, param in model.named_parameters():
        if "layers." not in name:
            continue
        if not any(name.endswith(suffix) for suffix in target_param_suffixes):
            continue
        target_names.append(name)
        all_magnitudes.append(param.data.abs().flatten().cpu())

    if len(all_magnitudes) == 0:
        raise ValueError("No target parameters found for pruning mask computation. "
                         "Expected parameters matching 'model.layers.*.weight'")

    all_magnitudes = torch.cat(all_magnitudes)
    N = all_magnitudes.numel()
    k = max(1, int(pruning_ratio * N))
    threshold = torch.kthvalue(all_magnitudes, k).values.item()
    print(f"[Ghost] Pruning ratio={pruning_ratio}, Total target params={N:,}, "
          f"k={k:,}, threshold τ={threshold:.6e}")

    mask_dict = {}
    m1_count = 0
    m0_count = 0
    for name, param in model.named_parameters():
        if name in target_names:
            mask = (param.data.abs() > threshold)
            mask_dict[name] = mask.cpu()
            m1_count += mask.sum().item()
            m0_count += (~mask).sum().item()

    print(f"[Ghost] M=1 (retained) params: {m1_count:,}, M=0 (ghost) params: {m0_count:,}")
    return mask_dict, threshold


class GhostPGDCallback(TrainerCallback):
    """Phase 2 回调：每步结束后对 M=0 位置执行 clamp(-τ, τ)，同时恢复 M=1 为原始值。"""

    def __init__(self, mask_dict, threshold, original_m1_values):
        self.mask_dict = mask_dict
        self.threshold = threshold
        self.original_m1_values = original_m1_values

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        assert model is not None

        with FSDP.summon_full_params(model, writeback=True):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    _name = name.replace("_fsdp_wrapped_module.", "").replace("module.", "").replace("_forward_module.", "")

                    if _name not in self.mask_dict:
                        continue

                    mask = self.mask_dict[_name].to(param.device)  # True=M=1
                    orig = self.original_m1_values[_name].to(param.device)

                    # 1. Restore M=1 positions to original values
                    param.data[mask] = orig[mask]

                    # 2. Clamp M=0 positions to [-τ, τ]
                    ghost_mask = ~mask
                    param.data[ghost_mask] = param.data[ghost_mask].clamp_(
                        -self.threshold, self.threshold
                    )
