from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .models import segment_mean


def _bond_graph(
    molecule: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only covalent edges, excluding 3D radius-only neighbors."""
    edge_features = molecule["edge_features"]
    mask = edge_features[:, :6].abs().sum(dim=-1) > 0
    return molecule["edge_index"][:, mask], edge_features[mask, :6]


def _segment_sum(
    values: torch.Tensor, index: torch.Tensor, size: int
) -> torch.Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, index, values)
    return output


def _segment_softmax(
    values: torch.Tensor, index: torch.Tensor, size: int
) -> torch.Tensor:
    if values.numel() == 0:
        return values
    shape = (size, *values.shape[1:])
    maximum = values.new_full(shape, -torch.inf)
    expanded = index.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    maximum.scatter_reduce_(
        0, expanded, values, reduce="amax", include_self=True
    )
    exponent = torch.exp(values - maximum[index])
    denominator = values.new_zeros(shape)
    denominator.scatter_add_(0, expanded, exponent)
    return exponent / denominator[index].clamp_min(1e-12)


class AtomInput(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.atomic_number = nn.Embedding(119, hidden_dim)
        self.features = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, molecule: dict[str, torch.Tensor]) -> torch.Tensor:
        atomic_number = molecule["atomic_numbers"].clamp(0, 118)
        return self.norm(
            self.atomic_number(atomic_number)
            + self.features(molecule["atom_features"])
        )


class PairSiteHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        atoms: torch.Tensor,
        atom_batch: torch.Tensor,
        other_graph: torch.Tensor,
    ) -> torch.Tensor:
        context = other_graph[atom_batch]
        features = torch.cat(
            [atoms, context, atoms * context, (atoms - context).abs()],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


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


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(6, heads, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atoms: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            attention_output = torch.zeros_like(atoms)
        else:
            receiver, sender = edge_index
            query = self.query(atoms).view(-1, self.heads, self.head_dim)
            key = self.key(atoms).view(-1, self.heads, self.head_dim)
            value = self.value(atoms).view(-1, self.heads, self.head_dim)
            score = (
                query[receiver] * key[sender]
            ).sum(dim=-1) / math.sqrt(self.head_dim)
            score = score + self.edge_bias(edge_features)
            attention = _segment_softmax(score, receiver, len(atoms))
            messages = attention.unsqueeze(-1) * value[sender]
            attention_output = _segment_sum(
                messages, receiver, len(atoms)
            ).reshape(len(atoms), -1)
        atoms = self.norm1(atoms + self.dropout(self.output(attention_output)))
        return self.norm2(atoms + self.dropout(self.ffn(atoms)))


class EAC2Hop(nn.Module):
    """Adapted Every-Atom-Counts local two-hop atom classifier."""

    baseline_variant = "adapted_eac_2hop"

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        cutoff: float = 5.0,
        **_: Any,
    ):
        super().__init__()
        del cutoff
        self.input = AtomInput(hidden_dim)
        self.local_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 12, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = PairSiteHead(hidden_dim, dropout)

    @staticmethod
    def _neighbor_mean(
        values: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.zeros_like(values)
        receiver, sender = edge_index
        output = _segment_sum(values[sender], receiver, len(values))
        degree = values.new_zeros(len(values))
        degree.index_add_(
            0,
            receiver,
            torch.ones(
                len(receiver), device=values.device, dtype=values.dtype
            ),
        )
        return output / degree.clamp_min(1).unsqueeze(-1)

    @staticmethod
    def _edge_mean(
        values: torch.Tensor, edge_index: torch.Tensor, atom_count: int
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return values.new_zeros((atom_count, values.shape[-1]))
        receiver = edge_index[0]
        output = _segment_sum(values, receiver, atom_count)
        degree = values.new_zeros(atom_count)
        degree.index_add_(
            0,
            receiver,
            torch.ones(
                len(receiver), device=values.device, dtype=values.dtype
            ),
        )
        return output / degree.clamp_min(1).unsqueeze(-1)

    def _encode(
        self, molecule: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.input(molecule)
        edge_index, edge_features = _bond_graph(molecule)
        first = self._neighbor_mean(atoms, edge_index)
        second = self._neighbor_mean(first, edge_index)
        bond_first = self._edge_mean(
            edge_features, edge_index, len(atoms)
        )
        bond_second = self._neighbor_mean(
            F.pad(bond_first, (0, atoms.shape[-1] - 6)), edge_index
        )[:, :6]
        local = self.local_projection(
            torch.cat(
                [atoms, first, second, bond_first, bond_second], dim=-1
            )
        )
        batch_size = len(molecule["ptr"]) - 1
        return local, segment_mean(
            local, molecule["batch"], batch_size
        )

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1, g1 = self._encode(d1)
        h2, g2 = self._encode(d2)
        return {
            "logits1": self.head(h1, d1["batch"], g2),
            "logits2": self.head(h2, d2["batch"], g1),
        }


class DRACONAtom(nn.Module):
    """Adapted DRACON-style disconnected graph attention classifier."""

    baseline_variant = "adapted_dracon_atom"

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        cutoff: float = 5.0,
        **_: Any,
    ):
        super().__init__()
        del cutoff
        self.input = AtomInput(hidden_dim)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(hidden_dim, attention_heads, dropout)
            for _ in range(layers)
        )
        self.pair_node = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.atom_pair = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(
        self, molecule: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.input(molecule)
        edge_index, edge_features = _bond_graph(molecule)
        for layer in self.layers:
            atoms = layer(atoms, edge_index, edge_features)
        graph = segment_mean(
            atoms, molecule["batch"], len(molecule["ptr"]) - 1
        )
        return atoms, graph

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1, g1 = self._encode(d1)
        h2, g2 = self._encode(d2)
        pair = self.pair_node(
            torch.cat([g1, g2, g1 * g2, (g1 - g2).abs()], dim=-1)
        )
        u1 = self.norm(
            h1
            + self.atom_pair(torch.cat([h1, pair[d1["batch"]]], dim=-1))
        )
        u2 = self.norm(
            h2
            + self.atom_pair(torch.cat([h2, pair[d2["batch"]]], dim=-1))
        )
        return {
            "logits1": self.head(u1).squeeze(-1),
            "logits2": self.head(u2).squeeze(-1),
        }


class RMechRPSite(nn.Module):
    """Adapted RMechRP site encoder with reaction-level contrastive output."""

    baseline_variant = "adapted_rmechrp_site"

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.1,
        projection_dim: int = 128,
        cutoff: float = 5.0,
        **_: Any,
    ):
        super().__init__()
        del cutoff
        self.input = AtomInput(hidden_dim)
        self.layers = nn.ModuleList(
            BondMessageLayer(hidden_dim, dropout, gin=True)
            for _ in range(layers)
        )
        self.reaction = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.head = PairSiteHead(hidden_dim, dropout)

    def _encode(
        self, molecule: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.input(molecule)
        edge_index, edge_features = _bond_graph(molecule)
        for layer in self.layers:
            atoms = layer(atoms, edge_index, edge_features)
        graph = segment_mean(
            atoms, molecule["batch"], len(molecule["ptr"]) - 1
        )
        return atoms, graph

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1, g1 = self._encode(d1)
        h2, g2 = self._encode(d2)
        reaction = self.reaction(
            torch.cat([g1, g2, g1 * g2, (g1 - g2).abs()], dim=-1)
        )
        return {
            "logits1": self.head(h1, d1["batch"], reaction),
            "logits2": self.head(h2, d2["batch"], reaction),
            "reaction_embedding": F.normalize(
                self.projection(reaction), dim=-1
            ),
        }


class AttentiveGRULayer(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int, dropout: float):
        super().__init__()
        self.attention = GraphAttentionLayer(
            hidden_dim, attention_heads, dropout
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atoms: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        message = self.attention(atoms, edge_index, edge_features)
        return self.norm(self.gru(message, atoms))


class ReactAIvateRAI(nn.Module):
    """Adapted ReactAIvate GAT-GRU model with a virtual reaction supernode."""

    baseline_variant = "adapted_reactaivate_rai"

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.1,
        num_reaction_classes: int = 11,
        cutoff: float = 5.0,
        **_: Any,
    ):
        super().__init__()
        del cutoff
        self.input = AtomInput(hidden_dim)
        self.layers = nn.ModuleList(
            AttentiveGRULayer(hidden_dim, attention_heads, dropout)
            for _ in range(layers)
        )
        self.supernode = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.supernode_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.atom_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.site = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_reaction_classes),
        )

    def _encode(
        self, molecule: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.input(molecule)
        edge_index, edge_features = _bond_graph(molecule)
        for layer in self.layers:
            atoms = layer(atoms, edge_index, edge_features)
        graph = segment_mean(
            atoms, molecule["batch"], len(molecule["ptr"]) - 1
        )
        return atoms, graph

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1, g1 = self._encode(d1)
        h2, g2 = self._encode(d2)
        initial = self.supernode(
            torch.cat([g1, g2, g1 * g2, (g1 - g2).abs()], dim=-1)
        )
        reaction = self.supernode_gru((g1 + g2) / 2.0, initial)
        u1 = self.atom_norm(
            self.atom_update(reaction[d1["batch"]], h1)
        )
        u2 = self.atom_norm(
            self.atom_update(reaction[d2["batch"]], h2)
        )
        return {
            "logits1": self.site(u1).squeeze(-1),
            "logits2": self.site(u2).squeeze(-1),
            "class_logits": self.classifier(reaction),
        }


def _pack_pair(
    h1: torch.Tensor,
    h2: torch.Tensor,
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    separator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    ptr1 = d1["ptr"].tolist()
    ptr2 = d2["ptr"].tolist()
    lengths = [
        (ptr1[i + 1] - ptr1[i]) + 1 + (ptr2[i + 1] - ptr2[i])
        for i in range(len(ptr1) - 1)
    ]
    padded = h1.new_zeros((len(lengths), max(lengths), h1.shape[-1]))
    padding_mask = torch.ones(
        (len(lengths), max(lengths)),
        dtype=torch.bool,
        device=h1.device,
    )
    locations: list[tuple[int, int, int]] = []
    for i, length in enumerate(lengths):
        n1 = ptr1[i + 1] - ptr1[i]
        n2 = ptr2[i + 1] - ptr2[i]
        padded[i, :n1] = h1[ptr1[i] : ptr1[i + 1]]
        padded[i, n1] = separator
        padded[i, n1 + 1 : n1 + 1 + n2] = h2[
            ptr2[i] : ptr2[i + 1]
        ]
        padding_mask[i, :length] = False
        locations.append((n1, n2, length))
    return padded, padding_mask, locations


class SPABAStyle77(nn.Module):
    """Safe atom-Transformer adaptation of SPABA head 7_7.

    This does not load the original chemical language model or monkey-patch
    torch.nn.functional. It must be reported as SPABA-style, not official
    SPABA reproduction.
    """

    baseline_variant = "adapted_spaba_style_7_7_no_clm_pretraining"

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 8,
        attention_heads: int = 8,
        dropout: float = 0.1,
        cutoff: float = 5.0,
        **_: Any,
    ):
        super().__init__()
        del cutoff
        if layers < 8:
            raise ValueError("SPABA-style 7_7 requires at least 8 layers")
        self.input = AtomInput(hidden_dim)
        self.segment = nn.Embedding(3, hidden_dim)
        self.separator = nn.Parameter(torch.zeros(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.site = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.separator, std=0.02)

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1 = self.input(d1) + self.segment.weight[0]
        h2 = self.input(d2) + self.segment.weight[1]
        separator = self.separator + self.segment.weight[2]
        packed, mask, locations = _pack_pair(h1, h2, d1, d2, separator)
        encoded = self.transformer(
            packed, src_key_padding_mask=mask
        )
        output1 = []
        output2 = []
        for index, (n1, n2, _) in enumerate(locations):
            output1.append(encoded[index, :n1])
            output2.append(encoded[index, n1 + 1 : n1 + 1 + n2])
        return {
            "logits1": self.site(torch.cat(output1)).squeeze(-1),
            "logits2": self.site(torch.cat(output2)).squeeze(-1),
        }


BASELINES: dict[str, type[nn.Module]] = {
    "dracon_atom": DRACONAtom,
    "rmechrp_site": RMechRPSite,
    "reactaivate_rai": ReactAIvateRAI,
    "eac_2hop": EAC2Hop,
    "spaba_style_7_7": SPABAStyle77,
}


def build_reaction_site_baseline(config: dict[str, Any]) -> nn.Module:
    arguments = dict(config)
    family = str(arguments.pop("family")).lower()
    try:
        model_type = BASELINES[family]
    except KeyError as exc:
        choices = ", ".join(sorted(BASELINES))
        raise ValueError(
            f"Unknown reaction-site baseline {family!r}; choose from {choices}"
        ) from exc
    return model_type(**arguments)
