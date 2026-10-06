from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def segment_mean(values: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, batch, values)
    counts = values.new_zeros(size)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=values.dtype))
    return output / counts.clamp_min(1).view(size, *([1] * (values.ndim - 1)))


def segment_sum(values: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, batch, values)
    return output


class VectorLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nch,hk->nck", vector, self.weight)


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = float(num_rbf) / cutoff
        self.cutoff = float(cutoff)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(
            -self.gamma * (distance.unsqueeze(-1) - self.centers) ** 2
        )
        envelope = 0.5 * (
            torch.cos(math.pi * distance / self.cutoff) + 1.0
        )
        envelope = torch.where(
            distance <= self.cutoff, envelope, torch.zeros_like(envelope)
        )
        return rbf * envelope.unsqueeze(-1)


class PaiNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, edge_dim: int):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim * 3)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.vector_mix = VectorLinear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        unit: torch.Tensor,
        radial: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        receiver, sender = edge_index
        filters = self.filter_net(torch.cat([radial, edge_features], dim=-1))
        messages = self.message(scalar[sender]) * filters
        msg_s, msg_dir, msg_vec = messages.chunk(3, dim=-1)
        edge_vector = (
            unit.unsqueeze(-1) * msg_dir.unsqueeze(1)
            + vector[sender] * msg_vec.unsqueeze(1)
        )
        aggregate_s = torch.zeros_like(scalar)
        aggregate_v = torch.zeros_like(vector)
        aggregate_s.index_add_(0, receiver, msg_s.to(aggregate_s.dtype))
        aggregate_v.index_add_(
            0,
            receiver,
            edge_vector.to(aggregate_v.dtype),
        )
        scalar = self.norm(scalar + aggregate_s)
        vector = vector + aggregate_v

        mixed = self.vector_mix(vector)
        invariant = torch.linalg.vector_norm(mixed, dim=1)
        update_s, update_v = self.update(
            torch.cat([scalar, invariant], dim=-1)
        ).chunk(2, dim=-1)
        scalar = self.norm(scalar + update_s)
        vector = vector + torch.sigmoid(update_v).unsqueeze(1) * mixed
        return scalar, vector


class PaiNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 5,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        atom_feature_dim: int = 7,
        edge_dim: int = 7,
        max_atomic_number: int = 118,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.atom_projection = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.layers = nn.ModuleList(
            PaiNNLayer(hidden_dim, num_rbf, edge_dim) for _ in range(layers)
        )

    def forward(self, molecule: dict[str, torch.Tensor]):
        z = molecule["atomic_numbers"].clamp(
            0, self.atom_embedding.num_embeddings - 1
        )
        scalar = self.atom_embedding(z) + self.atom_projection(
            molecule["atom_features"]
        )
        vector = scalar.new_zeros((len(scalar), 3, scalar.shape[-1]))
        receiver, sender = molecule["edge_index"]
        displacement = molecule["pos"][receiver] - molecule["pos"][sender]
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1e-8)
        unit = displacement / distance.unsqueeze(-1)
        radial = self.rbf(distance)
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                molecule["edge_index"],
                unit,
                radial,
                molecule["edge_features"],
            )
        return scalar, vector


@dataclass
class DCIAOutput:
    message1: torch.Tensor
    message2: torch.Tensor
    global12: torch.Tensor
    global21: torch.Tensor
    details: list[dict[str, torch.Tensor]] | None


class DCIA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        interaction_dim: int = 128,
        pair_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if interaction_dim % heads:
            raise ValueError("interaction_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = interaction_dim // heads
        self.invariant_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, interaction_dim),
            nn.SiLU(),
            nn.LayerNorm(interaction_dim),
        )
        self.role_embedding = nn.Parameter(torch.randn(2, interaction_dim) * 0.02)
        self.q12 = nn.Linear(interaction_dim, interaction_dim)
        self.k12 = nn.Linear(interaction_dim, interaction_dim)
        self.v12 = nn.Linear(interaction_dim, interaction_dim)
        self.q21 = nn.Linear(interaction_dim, interaction_dim)
        self.k21 = nn.Linear(interaction_dim, interaction_dim)
        self.v21 = nn.Linear(interaction_dim, interaction_dim)
        self.message1 = nn.Linear(interaction_dim, hidden_dim)
        self.message2 = nn.Linear(interaction_dim, hidden_dim)
        pair_input = interaction_dim * 4
        self.pair12 = nn.Sequential(
            nn.Linear(pair_input, pair_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim),
        )
        self.pair21 = nn.Sequential(
            nn.Linear(pair_input, pair_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim),
        )

    def _invariant(
        self, scalar: torch.Tensor, vector: torch.Tensor, role: int
    ) -> torch.Tensor:
        norms = torch.linalg.vector_norm(vector, dim=1)
        return self.invariant_projection(torch.cat([scalar, norms], dim=-1)) + (
            self.role_embedding[role]
        )

    def _reshape_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(len(value), self.heads, self.head_dim)

    def forward(
        self,
        scalar1: torch.Tensor,
        vector1: torch.Tensor,
        ptr1: torch.Tensor,
        scalar2: torch.Tensor,
        vector2: torch.Tensor,
        ptr2: torch.Tensor,
        *,
        keep1: torch.Tensor | None = None,
        keep2: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> DCIAOutput:
        z1_all = self._invariant(scalar1, vector1, 0)
        z2_all = self._invariant(scalar2, vector2, 1)
        q12_all = self._reshape_heads(self.q12(z1_all))
        k12_all = self._reshape_heads(self.k12(z2_all))
        v12_all = self._reshape_heads(self.v12(z2_all))
        q21_all = self._reshape_heads(self.q21(z2_all))
        k21_all = self._reshape_heads(self.k21(z1_all))
        v21_all = self._reshape_heads(self.v21(z1_all))
        attended1_all = z1_all.new_zeros(z1_all.shape)
        attended2_all = z2_all.new_zeros(z2_all.shape)
        active1_all = (
            keep1.bool()
            if keep1 is not None
            else torch.ones(len(z1_all), dtype=torch.bool, device=z1_all.device)
        )
        active2_all = (
            keep2.bool()
            if keep2 is not None
            else torch.ones(len(z2_all), dtype=torch.bool, device=z2_all.device)
        )
        globals12 = []
        globals21 = []
        details = [] if return_details else None
        pair_inputs12 = []
        pair_inputs21 = []
        pair_weights12 = []
        pair_weights21 = []
        pair_cell_counts = []
        denoms1 = []
        denoms2 = []

        batch_size = len(ptr1) - 1
        for batch_index in range(batch_size):
            s1, e1 = int(ptr1[batch_index]), int(ptr1[batch_index + 1])
            s2, e2 = int(ptr2[batch_index]), int(ptr2[batch_index + 1])
            z1 = z1_all[s1:e1]
            z2 = z2_all[s2:e2]
            mask1 = (
                keep1[s1:e1].bool()
                if keep1 is not None
                else torch.ones(len(z1), dtype=torch.bool, device=z1.device)
            )
            mask2 = (
                keep2[s2:e2].bool()
                if keep2 is not None
                else torch.ones(len(z2), dtype=torch.bool, device=z2.device)
            )

            q1 = q12_all[s1:e1]
            k2 = k12_all[s2:e2]
            v2 = v12_all[s2:e2]
            score12 = torch.einsum("ihd,jhd->hij", q1, k2) / math.sqrt(
                self.head_dim
            )
            score12 = score12.masked_fill(~mask2[None, None, :], -1e4)
            attention12 = torch.softmax(score12, dim=-1)
            attention12 = attention12 * mask1[None, :, None]
            message1 = torch.einsum(
                "hij,jhd->ihd", attention12, v2
            ).reshape(len(z1), -1)
            attended1_all[s1:e1] = message1

            q2 = q21_all[s2:e2]
            k1 = k21_all[s1:e1]
            v1 = v21_all[s1:e1]
            score21 = torch.einsum("jhd,ihd->hji", q2, k1) / math.sqrt(
                self.head_dim
            )
            score21 = score21.masked_fill(~mask1[None, None, :], -1e4)
            attention21 = torch.softmax(score21, dim=-1)
            attention21 = attention21 * mask2[None, :, None]
            message2 = torch.einsum(
                "hji,ihd->jhd", attention21, v1
            ).reshape(len(z2), -1)
            attended2_all[s2:e2] = message2

            z1_grid = z1[:, None, :].expand(-1, len(z2), -1)
            z2_grid = z2[None, :, :].expand(len(z1), -1, -1)
            pair_input12 = torch.cat(
                [
                    z1_grid,
                    z2_grid,
                    z1_grid * z2_grid,
                    torch.abs(z1_grid - z2_grid),
                ],
                dim=-1,
            )
            z2_rev = z2[:, None, :].expand(-1, len(z1), -1)
            z1_rev = z1[None, :, :].expand(len(z2), -1, -1)
            pair_input21 = torch.cat(
                [
                    z2_rev,
                    z1_rev,
                    z2_rev * z1_rev,
                    torch.abs(z2_rev - z1_rev),
                ],
                dim=-1,
            )
            mean12 = attention12.mean(dim=0)
            mean21 = attention21.mean(dim=0)
            denom1 = mask1.sum().clamp_min(1)
            denom2 = mask2.sum().clamp_min(1)
            if details is not None:
                pair12 = self.pair12(pair_input12)
                pair21 = self.pair21(pair_input21)
                globals12.append(
                    (pair12 * mean12.unsqueeze(-1)).sum(dim=(0, 1)) / denom1
                )
                globals21.append(
                    (pair21 * mean21.unsqueeze(-1)).sum(dim=(0, 1)) / denom2
                )
                details.append(
                    {
                        "pair12": pair12,
                        "pair21": pair21,
                        "attention12": mean12,
                        "attention21": mean21,
                        "denom1": denom1,
                        "denom2": denom2,
                    }
                )
            else:
                pair_inputs12.append(pair_input12.reshape(-1, pair_input12.shape[-1]))
                pair_inputs21.append(pair_input21.reshape(-1, pair_input21.shape[-1]))
                pair_weights12.append(mean12.reshape(-1))
                pair_weights21.append(mean21.reshape(-1))
                pair_cell_counts.append(len(z1) * len(z2))
                denoms1.append(denom1)
                denoms2.append(denom2)
        if details is None:
            encoded12 = self.pair12(torch.cat(pair_inputs12, dim=0))
            encoded21 = self.pair21(torch.cat(pair_inputs21, dim=0))
            cell_counts = torch.as_tensor(
                pair_cell_counts,
                device=encoded12.device,
                dtype=torch.long,
            )
            pair_batch = torch.repeat_interleave(
                torch.arange(batch_size, device=encoded12.device),
                cell_counts,
            )
            # Accumulate attention-weighted pair features in FP32. CUDA autocast
            # keeps the attention weights in FP32 while the pair MLP emits BF16.
            global12 = torch.zeros(
                (batch_size, encoded12.shape[-1]),
                dtype=torch.float32,
                device=encoded12.device,
            )
            global21 = torch.zeros(
                (batch_size, encoded21.shape[-1]),
                dtype=torch.float32,
                device=encoded21.device,
            )
            global12.index_add_(
                0,
                pair_batch,
                encoded12.float()
                * torch.cat(pair_weights12).float().unsqueeze(-1),
            )
            global21.index_add_(
                0,
                pair_batch,
                encoded21.float()
                * torch.cat(pair_weights21).float().unsqueeze(-1),
            )
            global12 = global12 / torch.stack(denoms1).unsqueeze(-1)
            global21 = global21 / torch.stack(denoms2).unsqueeze(-1)
        else:
            global12 = torch.stack(globals12)
            global21 = torch.stack(globals21)
        message1_all = self.message1(attended1_all) * active1_all[:, None]
        message2_all = self.message2(attended2_all) * active2_all[:, None]
        return DCIAOutput(
            message1=message1_all,
            message2=message2_all,
            global12=global12,
            global21=global21,
            details=details,
        )


@dataclass
class CRDMOutput:
    scalar: torch.Tensor
    vector: torch.Tensor
    delta_scalar: torch.Tensor
    delta_vector: torch.Tensor
    response: torch.Tensor


class CRDM(nn.Module):
    def __init__(self, hidden_dim: int, class_dim: int, num_classes: int):
        super().__init__()
        self.scalar_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.response_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, class_dim),
            nn.SiLU(),
            nn.Linear(class_dim, class_dim),
        )
        self.class_embedding = nn.Parameter(
            torch.randn(num_classes, class_dim) * 0.02
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        message: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> CRDMOutput:
        invariant_input = torch.cat([scalar, message], dim=-1)
        conditioned_scalar = self.norm(
            scalar
            + self.scalar_gate(invariant_input)
            * self.scalar_update(invariant_input)
        )
        vector_scale = 1.0 + 0.1 * torch.tanh(
            self.vector_gate(invariant_input)
        )
        conditioned_vector = vector * vector_scale.unsqueeze(1)
        delta_scalar = conditioned_scalar - scalar
        delta_vector = conditioned_vector - vector
        delta_vector_norm = torch.linalg.vector_norm(delta_vector, dim=1)
        rho = torch.sqrt(
            delta_scalar.square().sum(dim=-1)
            + delta_vector_norm.square().sum(dim=-1)
            + 1e-12
        ) / math.sqrt(delta_scalar.shape[-1] * 2)
        response_features = self.response_projection(
            torch.cat(
                [
                    scalar,
                    conditioned_scalar,
                    delta_scalar,
                    delta_vector_norm,
                ],
                dim=-1,
            )
        )
        class_gate = torch.sigmoid(
            response_features @ self.class_embedding.transpose(0, 1)
        )
        response = rho.unsqueeze(-1) * class_gate
        if keep is not None:
            mask = keep.to(scalar.dtype).unsqueeze(-1)
            conditioned_scalar = conditioned_scalar * mask
            conditioned_vector = conditioned_vector * mask.unsqueeze(1)
            delta_scalar = delta_scalar * mask
            delta_vector = delta_vector * mask.unsqueeze(1)
            response = response * mask
        return CRDMOutput(
            conditioned_scalar,
            conditioned_vector,
            delta_scalar,
            delta_vector,
            response,
        )


class DirectionalGatedBilinearFusion(nn.Module):
    """Fuse molecular and DCIA pair summaries with a gated bilinear residual.

    The two directions use separate low-rank projections so swapping the drug
    order does not force identical predictions. A zero-initialized ReZero-style
    scalar makes the initial output exactly equal to the M1 logits.
    """

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        rank: int,
        fusion_dim: int,
        num_classes: int,
        dropout: float,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("DGBF rank must be positive")
        if fusion_dim <= 0:
            raise ValueError("DGBF fusion_dim must be positive")

        self.mol1_to_12 = nn.Linear(hidden_dim, rank, bias=False)
        self.mol2_to_12 = nn.Linear(hidden_dim, rank, bias=False)
        self.pair12_projection = nn.Linear(pair_dim, rank, bias=False)
        self.mol2_to_21 = nn.Linear(hidden_dim, rank, bias=False)
        self.mol1_to_21 = nn.Linear(hidden_dim, rank, bias=False)
        self.pair21_projection = nn.Linear(pair_dim, rank, bias=False)
        self.bilinear12_norm = nn.LayerNorm(rank)
        self.bilinear21_norm = nn.LayerNorm(rank)

        fusion_input_dim = hidden_dim * 2 + pair_dim * 2 + rank * 2
        self.candidate = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.output = nn.Linear(fusion_dim, num_classes)
        self.residual_alpha = nn.Parameter(
            torch.tensor(float(alpha_init), dtype=torch.float32)
        )

    @property
    def residual_scale(self) -> torch.Tensor:
        return torch.tanh(self.residual_alpha)

    def forward(
        self,
        molecule1: torch.Tensor,
        molecule2: torch.Tensor,
        interaction12: torch.Tensor,
        interaction21: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bilinear12 = self.bilinear12_norm(
            torch.tanh(self.mol1_to_12(molecule1))
            * torch.tanh(self.mol2_to_12(molecule2))
            * torch.tanh(self.pair12_projection(interaction12))
        )
        bilinear21 = self.bilinear21_norm(
            torch.tanh(self.mol2_to_21(molecule2))
            * torch.tanh(self.mol1_to_21(molecule1))
            * torch.tanh(self.pair21_projection(interaction21))
        )
        fusion_input = torch.cat(
            [
                molecule1,
                molecule2,
                interaction12,
                interaction21,
                bilinear12,
                bilinear21,
            ],
            dim=-1,
        )
        fused = self.fusion_norm(
            self.gate(fusion_input) * self.candidate(fusion_input)
        )
        residual_logits = self.output(fused)
        return (
            base_logits + self.residual_scale * residual_logits,
            residual_logits,
        )


class AttentiveFPMolecularReadout(nn.Module):
    """AttentiveFP-style molecular-level attentive recurrent readout.

    This module adapts the molecule-level readout introduced by Xiong et al.
    (J. Med. Chem. 2020, DOI: 10.1021/acs.jmedchem.9b00959) to consume PaiNN
    scalar atom representations. It is permutation invariant within each
    molecule and does not mix molecular coordinate frames.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_timesteps: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_timesteps <= 0:
            raise ValueError("AttentiveFP num_timesteps must be positive")
        self.num_timesteps = int(num_timesteps)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.graph_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _attention_pool(
        values: torch.Tensor,
        scores: torch.Tensor,
        ptr: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = []
        weights_all = torch.zeros_like(scores)
        for graph_index in range(len(ptr) - 1):
            start = int(ptr[graph_index])
            end = int(ptr[graph_index + 1])
            local_scores = scores[start:end]
            if keep is not None:
                local_keep = keep[start:end].bool()
                local_scores = local_scores.masked_fill(~local_keep, -1e4)
            local_weights = torch.softmax(local_scores, dim=0)
            if keep is not None:
                local_weights = local_weights * local_keep.to(
                    local_weights.dtype
                )
                local_weights = local_weights / local_weights.sum().clamp_min(
                    1e-12
                )
            weights_all[start:end] = local_weights
            contexts.append(
                (local_weights.unsqueeze(-1) * values[start:end]).sum(dim=0)
            )
        return torch.stack(contexts), weights_all

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        keep: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(ptr) - 1
        projected_nodes = torch.tanh(self.node_projection(node_features))
        graph_state = torch.tanh(
            segment_mean(projected_nodes, batch, batch_size)
        )
        last_attention = projected_nodes.new_zeros(len(projected_nodes))

        for _ in range(self.num_timesteps):
            graph_query = self.graph_projection(graph_state)[batch]
            scores = self.attention(
                torch.cat([projected_nodes, graph_query], dim=-1)
            ).squeeze(-1)
            context, last_attention = self._attention_pool(
                projected_nodes,
                scores,
                ptr,
                keep,
            )
            context = torch.nn.functional.elu(
                self.context_projection(context)
            )
            graph_state = self.gru(
                self.dropout(context),
                graph_state,
            )
        graph_state = self.output_norm(graph_state)
        if return_attention:
            return graph_state, last_attention
        return graph_state


class MSANPredictionMLP(nn.Sequential):
    """Prediction MLP used by the official CIKM 2022 MSAN implementation."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        if num_layers < 2:
            raise ValueError("MSAN prediction MLP requires at least 2 layers")

        def block(input_dim: int, output_dim: int) -> list[nn.Module]:
            return [
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        modules = block(hidden_dim, hidden_dim * 2)
        for _ in range(1, num_layers - 1):
            modules.extend(block(hidden_dim * 2, hidden_dim * 2))
        modules.append(nn.Linear(hidden_dim * 2, hidden_dim))
        super().__init__(*modules)


class MSANSubstructureExtractor(nn.Module):
    """Transformer-like MSAN-SE with a fixed set of learnable patterns.

    Zhu et al., CIKM 2022, DOI: 10.1145/3511808.3557648.
    The attention normalization follows the authors' official implementation:
    for every atom, assignment probabilities are normalized across patterns.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_patterns: int = 60,
        residual: bool = False,
    ):
        super().__init__()
        if num_patterns <= 0:
            raise ValueError("MSAN num_patterns must be positive")
        self.num_patterns = int(num_patterns)
        self.patterns = nn.Parameter(
            torch.empty(1, self.num_patterns, hidden_dim)
        )
        nn.init.xavier_uniform_(self.patterns)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.residual = bool(residual)

    @staticmethod
    def _dense_nodes(
        node: torch.Tensor,
        ptr: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = (ptr[1:] - ptr[:-1]).tolist()
        dense = pad_sequence(
            torch.split(node, lengths),
            batch_first=True,
        )
        max_atoms = dense.shape[1]
        length_tensor = torch.as_tensor(
            lengths, device=node.device, dtype=torch.long
        )
        valid = (
            torch.arange(max_atoms, device=node.device)[None, :]
            < length_tensor[:, None]
        )
        if keep is not None:
            dense_keep = pad_sequence(
                torch.split(keep.bool(), lengths),
                batch_first=True,
                padding_value=False,
            )
            valid = valid & dense_keep
        return dense, valid

    def forward(
        self,
        node: torch.Tensor,
        ptr: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dense, valid = self._dense_nodes(node, ptr, keep)
        keys = self.key(dense)
        values = self.value(dense)
        queries = self.query(
            self.patterns.expand(len(dense), -1, -1)
        )
        scores = torch.matmul(
            queries, keys.transpose(-1, -2)
        ) / math.sqrt(node.shape[-1])
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e4)
        attention = torch.softmax(scores, dim=1)
        attention = attention * valid.unsqueeze(1).to(attention.dtype)
        extracted = queries + torch.matmul(attention, values)
        projected = torch.relu(self.output(extracted))
        if self.residual:
            extracted = extracted + projected
        else:
            extracted = projected
        assignment = attention.detach().argmax(dim=1)
        return extracted, assignment, attention, valid


class MSANSubstructureReadout(nn.Module):
    """MSAN-SE/SI/SD readout adapted to PaiNN + DCIA atom embeddings.

    SE extracts a fixed number of substructure vectors, SI constructs the
    pattern-to-pattern cosine-similarity matrix, and SD masks one assigned
    substructure with the paper's 50% training probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_patterns: int = 60,
        prediction_layers: int = 3,
        dropout: float = 0.5,
        substructure_drop_probability: float = 0.5,
        extractor_residual: bool = False,
    ):
        super().__init__()
        if not 0.0 <= substructure_drop_probability <= 1.0:
            raise ValueError(
                "MSAN substructure_drop_probability must be in [0, 1]"
            )
        self.num_patterns = int(num_patterns)
        self.substructure_drop_probability = float(
            substructure_drop_probability
        )
        self.extractor = MSANSubstructureExtractor(
            hidden_dim,
            num_patterns,
            residual=extractor_residual,
        )
        prediction_input = hidden_dim * 2 + num_patterns * num_patterns
        self.predictor = nn.Sequential(
            nn.Linear(prediction_input, hidden_dim),
            MSANPredictionMLP(
                hidden_dim,
                num_layers=prediction_layers,
                dropout=dropout,
            ),
            nn.Linear(hidden_dim, num_classes),
        )

    def _drop_substructure(
        self,
        node: torch.Tensor,
        ptr: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dropped = torch.zeros(
            len(node), dtype=torch.bool, device=node.device
        )
        if (
            not self.training
            or self.substructure_drop_probability == 0.0
            or torch.rand((), device=node.device)
            >= self.substructure_drop_probability
        ):
            return node, dropped

        with torch.no_grad():
            _, assignment, _, valid = self.extractor(
                node.detach(), ptr, keep
            )
            for graph_index in range(len(ptr) - 1):
                start = int(ptr[graph_index])
                end = int(ptr[graph_index + 1])
                local_valid = valid[graph_index, : end - start]
                local_assignment = assignment[
                    graph_index, : end - start
                ]
                counts = torch.bincount(
                    local_assignment[local_valid],
                    minlength=self.num_patterns,
                )
                candidates = torch.where(counts > 0)[0]
                if len(candidates) == 0:
                    continue
                selected = candidates[
                    torch.randint(
                        len(candidates), (), device=node.device
                    )
                ]
                local_drop = local_valid & (
                    local_assignment == selected
                )
                dropped[start:end] = local_drop
        augmented = node.masked_fill(dropped.unsqueeze(-1), 0.0)
        return augmented, dropped

    def _encode(
        self,
        node: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        node, dropped = self._drop_substructure(node, ptr, keep)
        batch_size = len(ptr) - 1
        global_graph = segment_sum(node, batch, batch_size)
        patterns, assignment, _, _ = self.extractor(
            node, ptr, keep
        )
        patterns = torch.nn.functional.normalize(
            patterns, dim=-1
        )
        return global_graph, patterns, assignment, dropped

    def forward(
        self,
        node1: torch.Tensor,
        batch1: torch.Tensor,
        ptr1: torch.Tensor,
        node2: torch.Tensor,
        batch2: torch.Tensor,
        ptr2: torch.Tensor,
        keep1: torch.Tensor | None = None,
        keep2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        global1, patterns1, assignment1, dropped1 = self._encode(
            node1, batch1, ptr1, keep1
        )
        global2, patterns2, assignment2, dropped2 = self._encode(
            node2, batch2, ptr2, keep2
        )
        similarity = torch.matmul(
            patterns1, patterns2.transpose(-1, -2)
        )
        logits = self.predictor(
            torch.cat(
                [global1, global2, similarity.flatten(start_dim=1)],
                dim=-1,
            )
        )
        return {
            "logits": logits,
            "global1": global1,
            "global2": global2,
            "patterns1": patterns1,
            "patterns2": patterns2,
            "similarity": similarity,
            "assignment1": assignment1,
            "assignment2": assignment2,
            "dropped1": dropped1,
            "dropped2": dropped2,
        }


def class_weighted_pool(
    values: torch.Tensor,
    response: torch.Tensor,
    ptr: torch.Tensor,
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    lengths = (ptr[1:] - ptr[:-1]).tolist()
    value_parts = torch.split(values, lengths)
    response_parts = torch.split(response, lengths)
    padded_values = pad_sequence(value_parts, batch_first=True)
    padded_response = pad_sequence(
        response_parts,
        batch_first=True,
        padding_value=-1e4,
    )
    max_atoms = padded_values.shape[1]
    length_tensor = torch.as_tensor(
        lengths,
        device=values.device,
        dtype=torch.long,
    )
    valid = (
        torch.arange(max_atoms, device=values.device)[None, :]
        < length_tensor[:, None]
    )
    if keep is not None:
        padded_keep = pad_sequence(
            torch.split(keep.bool(), lengths),
            batch_first=True,
            padding_value=False,
        )
        valid = valid & padded_keep
    scores = padded_response.masked_fill(~valid.unsqueeze(-1), -1e4)
    weights = torch.softmax(scores, dim=1) * valid.unsqueeze(-1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("bnk,bnh->bkh", weights, padded_values)


class DCIR(nn.Module):
    """Double-Molecular Conditional Interaction and Representation-Difference Network.

    DCIR combines a shared PaiNN encoder, DCIA, and CRDM. The two molecular
    coordinate frames are never combined; DCIA consumes only invariant scalar
    features and vector norms.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        painn_layers: int = 5,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        interaction_dim: int = 128,
        pair_dim: int = 128,
        class_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        ablation: str = "M3",
        dgbf_rank: int = 64,
        dgbf_dim: int = 128,
        dgbf_alpha_init: float = 0.0,
        attentivefp_timesteps: int = 2,
        msan_num_patterns: int = 60,
        msan_prediction_layers: int = 3,
        msan_dropout: float = 0.5,
        msan_substructure_drop_probability: float = 0.5,
        msan_extractor_residual: bool = False,
        msan_residual_alpha_init: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ablation = ablation.upper()
        if self.ablation not in {
            "M0",
            "M1",
            "M2",
            "M3",
            "A2",
            "A2_ATTENTIVEFP",
            "A2_MSAN",
            "A2_MSAN_RESIDUAL",
        }:
            raise ValueError(
                "ablation must be one of M0, M1, M2, M3, A2, "
                "A2_ATTENTIVEFP, A2_MSAN, A2_MSAN_RESIDUAL"
            )
        self.use_dcia = self.ablation in {
            "M1",
            "M3",
            "A2",
            "A2_ATTENTIVEFP",
            "A2_MSAN",
            "A2_MSAN_RESIDUAL",
        }
        self.use_crdm = self.ablation in {"M2", "M3"}
        self.use_dgbf = self.ablation == "A2"
        self.use_attentivefp = self.ablation == "A2_ATTENTIVEFP"
        self.use_msan = self.ablation in {
            "A2_MSAN",
            "A2_MSAN_RESIDUAL",
        }
        self.use_msan_residual = (
            self.ablation == "A2_MSAN_RESIDUAL"
        )
        self.encoder = PaiNNEncoder(
            hidden_dim=hidden_dim,
            layers=painn_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )
        self.dcia = DCIA(
            hidden_dim,
            interaction_dim,
            pair_dim,
            heads,
            dropout,
        )
        self.global_condition = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.crdm1 = CRDM(hidden_dim, class_dim, num_classes)
        self.crdm2 = CRDM(hidden_dim, class_dim, num_classes)
        self.class_embedding = nn.Parameter(
            torch.randn(num_classes, class_dim) * 0.02
        )
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, class_dim),
        )
        self.m0_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.pair_head12 = nn.Linear(pair_dim, num_classes, bias=False)
        self.pair_head21 = nn.Linear(pair_dim, num_classes, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        if self.use_dgbf:
            self.dgbf = DirectionalGatedBilinearFusion(
                hidden_dim=hidden_dim,
                pair_dim=pair_dim,
                rank=dgbf_rank,
                fusion_dim=dgbf_dim,
                num_classes=num_classes,
                dropout=dropout,
                alpha_init=dgbf_alpha_init,
            )
        if self.use_attentivefp:
            self.attentive_message_norm = nn.LayerNorm(hidden_dim)
            self.attentive_readout = AttentiveFPMolecularReadout(
                hidden_dim=hidden_dim,
                num_timesteps=attentivefp_timesteps,
                dropout=dropout,
            )
        if self.use_msan:
            self.msan_message_norm = nn.LayerNorm(hidden_dim)
            self.msan_readout = MSANSubstructureReadout(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_patterns=msan_num_patterns,
                prediction_layers=msan_prediction_layers,
                dropout=msan_dropout,
                substructure_drop_probability=(
                    msan_substructure_drop_probability
                ),
                extractor_residual=msan_extractor_residual,
            )
        if self.use_msan_residual:
            self.msan_residual_alpha = nn.Parameter(
                torch.tensor(
                    float(msan_residual_alpha_init),
                    dtype=torch.float32,
                )
            )
        self._freeze_inactive_modules()

    @property
    def msan_residual_scale(self) -> torch.Tensor:
        if not self.use_msan_residual:
            raise AttributeError(
                "MSAN residual scale exists only for A2_MSAN_RESIDUAL"
            )
        return torch.tanh(self.msan_residual_alpha)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_inactive_modules(self) -> None:
        """Exclude modules that do not participate in an ablation forward."""
        if not self.use_dcia:
            self._freeze(self.dcia)
            self._freeze(self.pair_head12)
            self._freeze(self.pair_head21)
        if not self.use_crdm:
            self._freeze(self.crdm1)
            self._freeze(self.crdm2)
            self.class_embedding.requires_grad_(False)
            self._freeze(self.context_projection)
            self._freeze(self.global_condition)
            if (
                self.use_dcia
                and not self.use_attentivefp
                and not self.use_msan
            ):
                self._freeze(self.dcia.message1)
                self._freeze(self.dcia.message2)
            if self.use_msan and not self.use_msan_residual:
                self._freeze(self.m0_head)
        else:
            self._freeze(self.m0_head)
            if self.use_dcia:
                self._freeze(self.global_condition)

    @staticmethod
    def _apply_keep(
        scalar: torch.Tensor,
        vector: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keep is None:
            return scalar, vector
        mask = keep.to(scalar.dtype).unsqueeze(-1)
        return scalar * mask, vector * mask.unsqueeze(1)

    def _global_messages(
        self,
        scalar1: torch.Tensor,
        vector1: torch.Tensor,
        mol1: dict,
        scalar2: torch.Tensor,
        vector2: torch.Tensor,
        mol2: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(mol1["ptr"]) - 1
        inv1 = torch.cat(
            [scalar1, torch.linalg.vector_norm(vector1, dim=1)], dim=-1
        )
        inv2 = torch.cat(
            [scalar2, torch.linalg.vector_norm(vector2, dim=1)], dim=-1
        )
        global1 = segment_mean(inv1, mol1["batch"], batch_size)
        global2 = segment_mean(inv2, mol2["batch"], batch_size)
        return (
            self.global_condition(global2)[mol1["batch"]],
            self.global_condition(global1)[mol2["batch"]],
        )

    def forward(
        self,
        mol1: dict[str, torch.Tensor],
        mol2: dict[str, torch.Tensor],
        *,
        keep1: torch.Tensor | None = None,
        keep2: torch.Tensor | None = None,
        return_explanations: bool = False,
    ) -> dict[str, Any]:
        scalar1, vector1 = self.encoder(mol1)
        scalar2, vector2 = self.encoder(mol2)
        scalar1, vector1 = self._apply_keep(scalar1, vector1, keep1)
        scalar2, vector2 = self._apply_keep(scalar2, vector2, keep2)
        batch_size = len(mol1["ptr"]) - 1

        dcia = None
        if self.use_dcia:
            dcia = self.dcia(
                scalar1,
                vector1,
                mol1["ptr"],
                scalar2,
                vector2,
                mol2["ptr"],
                keep1=keep1,
                keep2=keep2,
                return_details=return_explanations,
            )
            message1, message2 = dcia.message1, dcia.message2
            global12, global21 = dcia.global12, dcia.global21
        elif self.use_crdm:
            message1, message2 = self._global_messages(
                scalar1, vector1, mol1, scalar2, vector2, mol2
            )
            global12 = scalar1.new_zeros(
                (batch_size, self.pair_head12.in_features)
            )
            global21 = scalar1.new_zeros(
                (batch_size, self.pair_head21.in_features)
            )
        else:
            message1 = scalar1.new_zeros(scalar1.shape)
            message2 = scalar2.new_zeros(scalar2.shape)
            global12 = scalar1.new_zeros(
                (batch_size, self.pair_head12.in_features)
            )
            global21 = scalar1.new_zeros(
                (batch_size, self.pair_head21.in_features)
            )

        if self.use_crdm:
            cond1 = self.crdm1(scalar1, vector1, message1, keep1)
            cond2 = self.crdm2(scalar2, vector2, message2, keep2)
            response1, response2 = cond1.response, cond2.response
            pooled1 = class_weighted_pool(
                cond1.scalar, response1, mol1["ptr"], keep1
            )
            pooled2 = class_weighted_pool(
                cond2.scalar, response2, mol2["ptr"], keep2
            )
            delta1 = class_weighted_pool(
                cond1.delta_scalar, response1, mol1["ptr"], keep1
            )
            delta2 = class_weighted_pool(
                cond2.delta_scalar, response2, mol2["ptr"], keep2
            )
            context = self.context_projection(
                torch.cat([pooled1, pooled2, delta1, delta2], dim=-1)
            )
            context_logits = (
                context * self.class_embedding.unsqueeze(0)
            ).sum(dim=-1)
        else:
            msan = None
            if self.use_msan:
                msan_node1 = self.msan_message_norm(
                    scalar1 + message1
                )
                msan_node2 = self.msan_message_norm(
                    scalar2 + message2
                )
                msan = self.msan_readout(
                    msan_node1,
                    mol1["batch"],
                    mol1["ptr"],
                    msan_node2,
                    mol2["batch"],
                    mol2["ptr"],
                    keep1,
                    keep2,
                )
            if self.use_attentivefp:
                attentive1 = self.attentive_message_norm(
                    scalar1 + message1
                )
                attentive2 = self.attentive_message_norm(
                    scalar2 + message2
                )
                pooled1_base = self.attentive_readout(
                    attentive1,
                    mol1["batch"],
                    mol1["ptr"],
                    keep1,
                )
                pooled2_base = self.attentive_readout(
                    attentive2,
                    mol2["batch"],
                    mol2["ptr"],
                    keep2,
                )
            else:
                pooled1_base = segment_mean(
                    scalar1, mol1["batch"], batch_size
                )
                pooled2_base = segment_mean(
                    scalar2, mol2["batch"], batch_size
                )
            if self.use_msan and not self.use_msan_residual:
                pooled1_base = msan["global1"]
                pooled2_base = msan["global2"]
                context_logits = msan["logits"]
            else:
                pair_base = torch.cat(
                    [
                        pooled1_base,
                        pooled2_base,
                        pooled1_base * pooled2_base,
                        torch.abs(pooled1_base - pooled2_base),
                    ],
                    dim=-1,
                )
                context_logits = self.m0_head(pair_base)
            response1 = scalar1.new_zeros((len(scalar1), self.num_classes))
            response2 = scalar2.new_zeros((len(scalar2), self.num_classes))

        logits = context_logits + self.bias
        if self.use_dcia:
            logits = (
                logits
                + self.pair_head12(global12)
                + self.pair_head21(global21)
            )
        dgbf_residual = None
        if self.use_dgbf:
            logits, dgbf_residual = self.dgbf(
                pooled1_base,
                pooled2_base,
                global12,
                global21,
                logits,
            )
        msan_residual_logits = None
        if self.use_msan_residual:
            if msan is None:
                raise RuntimeError("MSAN residual output is unavailable")
            msan_residual_logits = msan["logits"]
            logits = (
                logits
                + self.msan_residual_scale * msan_residual_logits
            )

        result: dict[str, Any] = {
            "interaction_logits": logits,
            "d1_atom_response": response1,
            "d2_atom_response": response2,
            "d1_batch": mol1["batch"],
            "d2_batch": mol2["batch"],
        }
        if self.use_dgbf:
            result["dgbf_residual_logits"] = dgbf_residual
            result["dgbf_residual_scale"] = self.dgbf.residual_scale
        if self.use_msan and msan is not None:
            result["msan_similarity"] = msan["similarity"]
            result["msan_assignment1"] = msan["assignment1"]
            result["msan_assignment2"] = msan["assignment2"]
            result["msan_dropped1"] = msan["dropped1"]
            result["msan_dropped2"] = msan["dropped2"]
        if self.use_msan_residual:
            result["msan_residual_logits"] = msan_residual_logits
            result["msan_residual_scale"] = self.msan_residual_scale
        if return_explanations and dcia is not None and dcia.details is not None:
            contributions12 = []
            contributions21 = []
            for detail in dcia.details:
                c12 = torch.einsum(
                    "ijp,kp->kij",
                    detail["pair12"],
                    self.pair_head12.weight,
                )
                c12 = (
                    c12
                    * detail["attention12"].unsqueeze(0)
                    / detail["denom1"]
                )
                c21 = torch.einsum(
                    "jip,kp->kji",
                    detail["pair21"],
                    self.pair_head21.weight,
                )
                c21 = (
                    c21
                    * detail["attention21"].unsqueeze(0)
                    / detail["denom2"]
                )
                contributions12.append(c12)
                contributions21.append(c21)
            result["cross_atom_contributions_12"] = contributions12
            result["cross_atom_contributions_21"] = contributions21
        return result

    @torch.no_grad()
    def predict(
        self,
        mol1: dict[str, torch.Tensor],
        mol2: dict[str, torch.Tensor],
        task_mode: str,
        thresholds: torch.Tensor | None = None,
        return_explanations: bool = True,
    ) -> dict[str, Any]:
        output = self(
            mol1, mol2, return_explanations=return_explanations
        )
        logits = output["interaction_logits"]
        if task_mode == "multiclass":
            probability = torch.softmax(logits, dim=-1)
            predicted = probability.argmax(dim=-1)
            confidence = probability.max(dim=-1).values
        else:
            probability = torch.sigmoid(logits)
            if thresholds is None:
                thresholds = probability.new_full((self.num_classes,), 0.5)
            predicted = probability >= thresholds
            confidence = torch.abs(probability - thresholds).mean(dim=-1)
        output.update(
            {
                "interaction_probability": probability,
                "predicted_type": predicted,
                "prediction_confidence": confidence,
            }
        )
        return output
