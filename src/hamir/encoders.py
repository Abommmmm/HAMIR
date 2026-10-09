from __future__ import annotations

import math

import torch
from torch import nn


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
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
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
        rbf = torch.exp(-self.gamma * (distance.unsqueeze(-1) - self.centers) ** 2)
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
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
        edge_vector = unit.unsqueeze(-1) * msg_dir.unsqueeze(1) + vector[
            sender
        ] * msg_vec.unsqueeze(1)
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
        update_s, update_v = self.update(torch.cat([scalar, invariant], dim=-1)).chunk(
            2, dim=-1
        )
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
        z = molecule["atomic_numbers"].clamp(0, self.atom_embedding.num_embeddings - 1)
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
