from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    task_mode: str,
    class_weight: torch.Tensor | None = None,
    loss_name: str = "weighted",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    # Keep loss reductions in FP32 even when the model forward uses BF16.
    logits = logits.float()
    target = target.float() if task_mode != "multiclass" else target
    class_weight = (
        class_weight.float() if class_weight is not None else None
    )
    loss_name = loss_name.lower()
    if task_mode == "multiclass":
        base = F.cross_entropy(
            logits,
            target,
            weight=class_weight,
            reduction="none",
        )
        if loss_name == "focal":
            probability = torch.softmax(logits, dim=-1)
            pt = probability.gather(1, target.unsqueeze(1)).squeeze(1)
            base = (1.0 - pt).pow(focal_gamma) * base
        return base.mean()

    positive_weight = class_weight
    base = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
        reduction="none",
    )
    if loss_name == "focal":
        probability = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probability, 1.0 - probability)
        base = (1.0 - pt).pow(focal_gamma) * base
    return base.mean()


def js_divergence_from_logits(
    logits_a: torch.Tensor, logits_b: torch.Tensor, task_mode: str
) -> torch.Tensor:
    logits_a = logits_a.float()
    logits_b = logits_b.float()
    eps = 1e-6
    if task_mode == "multiclass":
        p = torch.softmax(logits_a, dim=-1).clamp(min=eps, max=1.0)
        q = torch.softmax(logits_b, dim=-1).clamp(min=eps, max=1.0)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
    else:
        # Saturated sigmoid outputs can be exactly 0 or 1. Passing those as
        # KL targets gives a finite value but a NaN derivative through
        # x * log(x), so keep both Bernoulli outcomes strictly inside (0, 1).
        p = torch.sigmoid(logits_a).clamp(min=eps, max=1.0 - eps)
        q = torch.sigmoid(logits_b).clamp(min=eps, max=1.0 - eps)
        p = torch.stack([p, 1.0 - p], dim=-1)
        q = torch.stack([q, 1.0 - q], dim=-1)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(midpoint.log(), p, reduction="batchmean")
        + F.kl_div(midpoint.log(), q, reduction="batchmean")
    )


def response_sparsity(
    response1: torch.Tensor, response2: torch.Tensor
) -> torch.Tensor:
    response1 = response1.float()
    response2 = response2.float()
    return 0.5 * (response1.mean() + response2.mean())


def faithfulness_margin(
    logits: torch.Tensor,
    masked_logits: torch.Tensor,
    target: torch.Tensor,
    task_mode: str,
    margin: float = 0.1,
) -> torch.Tensor:
    logits = logits.float()
    masked_logits = masked_logits.float()
    target = target.float() if task_mode != "multiclass" else target
    if task_mode == "multiclass":
        original = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        masked = masked_logits.gather(1, target.unsqueeze(1)).squeeze(1)
        drop = original - masked
    else:
        positive_count = target.sum(dim=-1).clamp_min(1)
        drop = ((logits - masked_logits) * target).sum(dim=-1) / positive_count
    return F.relu(margin - drop).mean()


def build_top_response_keep_masks(
    output: dict,
    target: torch.Tensor,
    ptr1: torch.Tensor,
    ptr2: torch.Tensor,
    task_mode: str,
    fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    response1 = output["d1_atom_response"].detach()
    response2 = output["d2_atom_response"].detach()
    keep1 = torch.ones(len(response1), dtype=torch.bool, device=response1.device)
    keep2 = torch.ones(len(response2), dtype=torch.bool, device=response2.device)
    for batch_index in range(len(ptr1) - 1):
        if task_mode == "multiclass":
            label = int(target[batch_index])
            score1 = response1[
                int(ptr1[batch_index]) : int(ptr1[batch_index + 1]), label
            ]
            score2 = response2[
                int(ptr2[batch_index]) : int(ptr2[batch_index + 1]), label
            ]
        else:
            labels = torch.where(target[batch_index] > 0.5)[0]
            score1 = response1[
                int(ptr1[batch_index]) : int(ptr1[batch_index + 1])
            ][:, labels].mean(dim=-1)
            score2 = response2[
                int(ptr2[batch_index]) : int(ptr2[batch_index + 1])
            ][:, labels].mean(dim=-1)
        for score, keep, ptr in (
            (score1, keep1, ptr1),
            (score2, keep2, ptr2),
        ):
            count = max(1, round(len(score) * fraction))
            count = min(count, max(len(score) - 1, 0))
            if count == 0:
                continue
            local_top = torch.topk(score, count).indices
            start = int(ptr[batch_index])
            keep[start + local_top] = False
    return keep1, keep2
