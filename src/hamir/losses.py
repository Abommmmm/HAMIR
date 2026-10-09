from __future__ import annotations
import torch
import torch.nn.functional as F


def _supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.new_zeros(())
    embedding = F.normalize(embedding.float(), dim=-1)
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return embedding.new_zeros(())
    similarity = similarity.masked_fill(identity, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    per_example = -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )
    return per_example[valid].mean().to(embedding.dtype)


def site_loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    *,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    dice_weight: float,
    group_pos_weight: torch.Tensor | None = None,
    group_weight: float = 0.0,
    consistency_weight: float = 0.0,
    class_weight: float = 0.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
) -> torch.Tensor:
    logits = torch.cat([output["logits1"], output["logits2"]])
    target = torch.cat([batch["target1"], batch["target2"]])
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    focal = ((1.0 - pt).pow(focal_gamma) * bce).mean()
    intersection = (probability * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + target.sum() + 1.0)
    loss = focal + dice_weight * dice
    if group_weight and "group_logits1" in output:
        group_logits = torch.cat([output["group_logits1"], output["group_logits2"]])
        group_target = torch.cat([batch["motif_target1"], batch["motif_target2"]])
        group_loss = F.binary_cross_entropy_with_logits(
            group_logits,
            group_target,
            pos_weight=group_pos_weight,
        )
        loss = loss + group_weight * group_loss
    if consistency_weight and "group_logits1" in output:
        consistency = logits.new_zeros(())
        for role in ("1", "2"):
            atom_probability = torch.sigmoid(output[f"logits{role}"])
            motif_index = batch[f"d{role}"]["motif_index"]
            group_probability = torch.sigmoid(output[f"group_logits{role}"])
            atom_group_probability = torch.zeros_like(group_probability)
            atom_group_probability.scatter_reduce_(
                0,
                motif_index,
                atom_probability,
                reduce="amax",
                include_self=True,
            )
            consistency = consistency + F.mse_loss(
                group_probability, atom_group_probability
            )
        loss = loss + consistency_weight * consistency / 2.0
    if class_weight and "class_logits" in output:
        class_logits = output["class_logits"]
        reaction_class = batch["reaction_class"]
        if int(reaction_class.max()) >= class_logits.shape[-1]:
            raise ValueError("reaction_class exceeds the configured number of classes")
        loss = loss + class_weight * F.cross_entropy(class_logits, reaction_class)
    if contrastive_weight and "reaction_embedding" in output:
        contrastive = _supervised_contrastive_loss(
            output["reaction_embedding"],
            batch["reaction_class"],
            contrastive_temperature,
        )
        loss = loss + contrastive_weight * contrastive
    return loss
