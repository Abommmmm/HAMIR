from __future__ import annotations

import torch
from torch import nn

def _segment_sum(
    values: torch.Tensor, index: torch.Tensor, size: int
) -> torch.Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, index, values)
    return output

class BondMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, *, gin: bool = False):
        super().__init__()
        self.gin = gin
        self.edge = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.epsilon = nn.Parameter(torch.zeros(())) if gin else None

    def forward(
        self,
        atoms: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            aggregate = torch.zeros_like(atoms)
        else:
            receiver, sender = edge_index
            messages = self.message(
                torch.cat([atoms[sender], self.edge(edge_features)], dim=-1)
            )
            aggregate = _segment_sum(messages, receiver, len(atoms))
        if self.gin:
            updated = (1.0 + self.epsilon) * atoms + aggregate
        else:
            degree = atoms.new_zeros(len(atoms))
            if edge_index.shape[1]:
                degree.index_add_(
                    0,
                    edge_index[0],
                    torch.ones(
                        edge_index.shape[1],
                        device=atoms.device,
                        dtype=atoms.dtype,
                    ),
                )
            updated = atoms + aggregate / degree.clamp_min(1).unsqueeze(-1)
        return self.norm(atoms + self.update(updated))
