"""Shared molecular encoding, sparse atom attention and BRICS motif feedback."""

from __future__ import annotations
import torch
from torch import nn
from .encoders import PaiNNEncoder, segment_mean


class HAMIR(nn.Module):
    """Hierarchical atom-motif interaction model for two-reactant site prediction."""

    def __init__(
        self,
        hidden_dim: int = 128,
        painn_layers: int = 5,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        dropout: float = 0.1,
        architecture: str = "hierarchical_motif",
        attention_heads: int = 4,
        attention_topk: int = 4,
        encoder_type: str = "painn",
    ):
        super().__init__()
        if architecture != "hierarchical_motif":
            raise ValueError(
                "The source-only release supports only the HAMIR hierarchical_motif architecture."
            )
        if (
            attention_heads < 1
            or attention_topk < 1
            or hidden_dim < 2
            or painn_layers < 1
            or num_rbf < 1
            or cutoff <= 0
        ):
            raise ValueError(
                "Model dimensions, attention_heads, attention_topk, layers and cutoff must be positive"
            )
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.architecture = architecture
        self.attention_topk = int(attention_topk)
        if encoder_type not in {"painn", "bond_mpnn"}:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.baseline_variant = (
            "hamir_2d" if encoder_type == "bond_mpnn" else "hamir_3d"
        )
        self.encoder = PaiNNEncoder(
            hidden_dim=hidden_dim, layers=painn_layers, num_rbf=num_rbf, cutoff=cutoff
        )
        self.invariant = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        if encoder_type == "bond_mpnn":
            from .encoder_2d import BondEncoder

            self.encoder = BondEncoder(hidden_dim, painn_layers)
            self.invariant = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
            )
            self.baseline_variant = "hamir_2d"
        self.site_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_scale_raw = nn.Parameter(torch.tensor(-2.0))
        self.motif_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.motif_norm = nn.LayerNorm(hidden_dim)
        self.motif_scale_raw = nn.Parameter(torch.tensor(-2.0))
        self.motif_to_atom = nn.Linear(hidden_dim, hidden_dim)
        self.atom_motif_norm = nn.LayerNorm(hidden_dim)
        self.group_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode(self, molecule: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.encoder_type == "bond_mpnn":
            embedding = self.invariant(self.encoder(molecule))
        else:
            scalar, vector = self.encoder(molecule)
            vector_norm = torch.linalg.vector_norm(vector, dim=1)
            embedding = self.invariant(torch.cat([scalar, vector_norm], dim=-1))
        occlusion_mask = molecule.get("occlusion_mask")
        if occlusion_mask is not None:
            embedding = embedding.clone()
            atom_batch = molecule["batch"]
            for index in range(int(len(molecule["ptr"]) - 1)):
                local = atom_batch == index
                masked = local & occlusion_mask
                available = local & ~occlusion_mask
                if not bool(masked.any()):
                    continue
                source = embedding[available]
                if not len(source):
                    source = embedding[local]
                replacement = source.mean(dim=0)
                embedding[masked] = replacement
        return embedding

    def _conditioned_logits(
        self,
        atoms: torch.Tensor,
        atom_batch: torch.Tensor,
        other_global: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        context = other_global[atom_batch]
        features = torch.cat(
            [atoms, context, atoms * context, torch.abs(atoms - context)], dim=-1
        )
        if deterministic:
            output = features
            for layer in self.site_head:
                if not isinstance(layer, nn.Dropout):
                    output = layer(output)
            return output.squeeze(-1)
        return self.site_head(features).squeeze(-1)

    def _sparse_bidirectional_attention(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        d1: dict,
        d2: dict,
        preliminary1: torch.Tensor,
        preliminary2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated1 = h1.clone()
        updated2 = h2.clone()
        ptr1 = d1["ptr"].tolist()
        ptr2 = d2["ptr"].tolist()
        for index in range(len(ptr1) - 1):
            start1, end1 = (ptr1[index], ptr1[index + 1])
            start2, end2 = (ptr2[index], ptr2[index + 1])
            count1 = min(self.attention_topk, end1 - start1)
            count2 = min(self.attention_topk, end2 - start2)
            local1 = torch.topk(preliminary1[start1:end1], k=count1).indices + start1
            local2 = torch.topk(preliminary2[start2:end2], k=count2).indices + start2
            query1 = h1[local1].unsqueeze(0)
            query2 = h2[local2].unsqueeze(0)
            message1, _ = self.cross_attention(
                query1, query2, query2, need_weights=False
            )
            message2, _ = self.cross_attention(
                query2, query1, query1, need_weights=False
            )
            scale = torch.sigmoid(self.cross_scale_raw)
            refined1 = self.cross_norm(h1[local1] + scale * message1.squeeze(0))
            refined2 = self.cross_norm(h2[local2] + scale * message2.squeeze(0))
            updated1.index_copy_(0, local1, refined1)
            updated2.index_copy_(0, local2, refined2)
        return (updated1, updated2)

    def _hierarchical_motif_update(
        self, h1: torch.Tensor, h2: torch.Tensor, d1: dict, d2: dict, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motif1 = segment_mean(h1, d1["motif_index"], int(len(d1["motif_batch"])))
        motif2 = segment_mean(h2, d2["motif_index"], int(len(d2["motif_batch"])))
        updated1 = motif1.clone()
        updated2 = motif2.clone()
        scale = torch.sigmoid(self.motif_scale_raw)
        for index in range(batch_size):
            local1 = torch.nonzero(d1["motif_batch"] == index, as_tuple=False).flatten()
            local2 = torch.nonzero(d2["motif_batch"] == index, as_tuple=False).flatten()
            query1 = motif1[local1].unsqueeze(0)
            query2 = motif2[local2].unsqueeze(0)
            message1, _ = self.motif_attention(
                query1, query2, query2, need_weights=False
            )
            message2, _ = self.motif_attention(
                query2, query1, query1, need_weights=False
            )
            updated1.index_copy_(
                0, local1, self.motif_norm(motif1[local1] + scale * message1.squeeze(0))
            )
            updated2.index_copy_(
                0, local2, self.motif_norm(motif2[local2] + scale * message2.squeeze(0))
            )
        h1 = self.atom_motif_norm(
            h1 + scale * self.motif_to_atom(updated1[d1["motif_index"]])
        )
        h2 = self.atom_motif_norm(
            h2 + scale * self.motif_to_atom(updated2[d2["motif_index"]])
        )
        return (
            h1,
            h2,
            self.group_head(updated1).squeeze(-1),
            self.group_head(updated2).squeeze(-1),
        )

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1 = self._encode(d1)
        h2 = self._encode(d2)
        batch_size = int(len(d1["ptr"]) - 1)
        g1 = segment_mean(h1, d1["batch"], batch_size)
        g2 = segment_mean(h2, d2["batch"], batch_size)
        preliminary1 = self._conditioned_logits(h1, d1["batch"], g2, deterministic=True)
        preliminary2 = self._conditioned_logits(h2, d2["batch"], g1, deterministic=True)
        result: dict[str, torch.Tensor] = {}
        h1, h2 = self._sparse_bidirectional_attention(
            h1, h2, d1, d2, preliminary1, preliminary2
        )
        result["preliminary_logits1"] = preliminary1
        result["preliminary_logits2"] = preliminary2
        h1, h2, group1, group2 = self._hierarchical_motif_update(
            h1, h2, d1, d2, batch_size
        )
        result["group_logits1"] = group1
        result["group_logits2"] = group2
        g1 = segment_mean(h1, d1["batch"], batch_size)
        g2 = segment_mean(h2, d2["batch"], batch_size)
        if getattr(self, "return_embeddings", False):
            result["reaction_embedding"] = torch.cat(
                [g1, g2, torch.abs(g1 - g2), g1 * g2], dim=-1
            )
        result.update(
            {
                "logits1": self._conditioned_logits(h1, d1["batch"], g2),
                "logits2": self._conditioned_logits(h2, d2["batch"], g1),
            }
        )
        return result
