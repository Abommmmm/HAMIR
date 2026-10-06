from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import load_config
from .deepddi_baseline import (
    PairExample,
    file_sha256,
    load_pair_examples,
    macro_f1_at_half,
)
from .losses import classification_loss
from .train_cloud import (
    assert_cloud_training_authorized,
    load_audit,
    set_seed,
)


ATOM_SYMBOLS = (
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
)
HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
ATOM_FEATURE_DIM = 55
GRAPH_CACHE_VERSION = 1


def one_hot_unknown(value, allowable_values: tuple) -> list[float]:
    if value not in allowable_values:
        value = allowable_values[-1]
    return [float(value == candidate) for candidate in allowable_values]


def atom_features(atom: Chem.Atom) -> torch.Tensor:
    """The 55-dimensional atom vector used by SSI-DDI and DSN-DDI."""
    try:
        implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
    except AttributeError:
        implicit_valence = atom.GetImplicitValence()
    values = (
        one_hot_unknown(atom.GetSymbol(), ATOM_SYMBOLS)
        + [
            atom.GetDegree() / 10.0,
            float(implicit_valence),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
        ]
        + one_hot_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
        ]
    )
    if len(values) != ATOM_FEATURE_DIM:
        raise AssertionError(
            f"Expected {ATOM_FEATURE_DIM} atom features, got {len(values)}"
        )
    return torch.tensor(values, dtype=torch.float32)


def molecule_graph_from_smiles(smiles: str) -> dict[str, torch.Tensor]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    features = torch.stack([atom_features(atom) for atom in molecule.GetAtoms()])
    edges = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
    ]
    if edges:
        directed = edges + [(target, source) for source, target in edges]
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return {"x": features, "edge_index": edge_index}


def drug_ids_from_records(path: Path) -> list[str]:
    drug_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            drug_ids.update((row["d1"].strip(), row["d2"].strip()))
    if not drug_ids:
        raise ValueError(f"No drug IDs found in {path}")
    return sorted(drug_ids)


def prepare_graph_cache(config: dict, force: bool = False) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    cache_path = paths["graph_cache"]
    if cache_path.exists() and not force:
        cache = load_graph_cache(cache_path)
        expected = set(drug_ids_from_records(paths["records"]))
        if expected.issubset(cache):
            print(
                f"Graph cache already covers all {len(expected)} drugs: "
                f"{cache_path}"
            )
            return
        raise ValueError(
            f"Existing graph cache is incomplete: {cache_path}; "
            "rerun prepare with --force"
        )

    graphs: dict[str, dict[str, torch.Tensor]] = {}
    drug_ids = drug_ids_from_records(paths["records"])
    conformer_dir = paths["conformers"]
    for drug_id in tqdm(
        drug_ids, desc="prepare molecular graphs", dynamic_ncols=True
    ):
        conformer_path = conformer_dir / f"{drug_id}.npz"
        if not conformer_path.is_file():
            raise FileNotFoundError(
                f"Missing conformer archive for {drug_id}: {conformer_path}"
            )
        with np.load(conformer_path, allow_pickle=False) as archive:
            smiles = str(archive["canonical_smiles"].item())
        graphs[drug_id] = molecule_graph_from_smiles(smiles)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "version": GRAPH_CACHE_VERSION,
            "feature_dim": ATOM_FEATURE_DIM,
            "graphs": graphs,
        },
        temporary_path,
    )
    temporary_path.replace(cache_path)
    print(f"Saved {len(graphs)} molecular graphs to {cache_path}")


def load_graph_cache(
    path: Path,
) -> dict[str, dict[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing graph cache: {path}. Run the prepare command first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != GRAPH_CACHE_VERSION:
        raise ValueError(
            f"Unsupported graph cache version in {path}: "
            f"{payload.get('version')}"
        )
    if int(payload.get("feature_dim", -1)) != ATOM_FEATURE_DIM:
        raise ValueError(f"Unexpected atom feature dimension in {path}")
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError(f"No molecular graphs in {path}")
    return graphs


class GraphPairDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        num_classes: int,
    ) -> None:
        self.examples = examples
        self.targets = torch.zeros(
            (len(examples), num_classes), dtype=torch.float32
        )
        for index, example in enumerate(examples):
            self.targets[index, list(example.labels)] = 1.0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> int:
        return index


@dataclass
class MoleculeBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    graph_count: int

    def to(self, device: torch.device) -> MoleculeBatch:
        return MoleculeBatch(
            x=self.x.to(device, non_blocking=True),
            edge_index=self.edge_index.to(device, non_blocking=True),
            batch=self.batch.to(device, non_blocking=True),
            graph_count=self.graph_count,
        )


def batch_molecular_graphs(
    graphs: list[dict[str, torch.Tensor]],
) -> MoleculeBatch:
    features = []
    edges = []
    membership = []
    offset = 0
    for graph_index, graph in enumerate(graphs):
        count = len(graph["x"])
        features.append(graph["x"])
        edges.append(graph["edge_index"] + offset)
        membership.append(
            torch.full((count,), graph_index, dtype=torch.long)
        )
        offset += count
    return MoleculeBatch(
        x=torch.cat(features),
        edge_index=torch.cat(edges, dim=1),
        batch=torch.cat(membership),
        graph_count=len(graphs),
    )


def segment_sum(
    values: torch.Tensor, index: torch.Tensor, count: int
) -> torch.Tensor:
    output = values.new_zeros((count,) + values.shape[1:])
    output.index_add_(0, index, values)
    return output


def segment_softmax(
    values: torch.Tensor, index: torch.Tensor, count: int
) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    expanded_index = index.view(-1, *([1] * (values.ndim - 1))).expand_as(
        values
    )
    maxima = values.new_full((count,) + values.shape[1:], -torch.inf)
    maxima.scatter_reduce_(
        0, expanded_index, values, reduce="amax", include_self=True
    )
    exponent = torch.exp(values - maxima[index])
    denominator = segment_sum(exponent, index, count).clamp_min(1e-12)
    result = exponent / denominator[index]
    return result.squeeze(-1) if squeeze else result


class GraphLayerNorm(nn.Module):
    """PyTorch equivalent of PyG LayerNorm(mode='graph')."""

    def __init__(self, feature_dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(
        self,
        features: torch.Tensor,
        membership: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        counts = torch.bincount(
            membership, minlength=graph_count
        ).to(features.dtype)
        denominators = (counts * self.feature_dim).clamp_min(1.0)
        sums = segment_sum(
            features.sum(dim=-1), membership, graph_count
        )
        means = sums / denominators
        centered = features - means[membership].unsqueeze(-1)
        square_sums = segment_sum(
            centered.square().sum(dim=-1), membership, graph_count
        )
        variances = square_sums / denominators
        normalized = centered / torch.sqrt(
            variances[membership].unsqueeze(-1) + self.epsilon
        )
        return normalized * self.weight + self.bias


class GraphAttentionConv(nn.Module):
    """Native PyTorch reproduction of the GATConv used by both papers."""

    def __init__(
        self,
        input_dim: int | tuple[int, int],
        output_dim: int,
        heads: int,
        add_self_loops: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        bipartite = isinstance(input_dim, tuple)
        if bipartite:
            source_dim, target_dim = input_dim
        else:
            source_dim = target_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        self.source_projection = nn.Linear(
            source_dim, output_dim * heads, bias=False
        )
        if not bipartite:
            self.target_projection = self.source_projection
        else:
            self.target_projection = nn.Linear(
                target_dim, output_dim * heads, bias=False
            )
        self.attention_source = nn.Parameter(
            torch.empty(1, heads, output_dim)
        )
        self.attention_target = nn.Parameter(
            torch.empty(1, heads, output_dim)
        )
        self.bias = nn.Parameter(torch.zeros(heads * output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.source_projection.weight)
        if self.target_projection is not self.source_projection:
            nn.init.xavier_uniform_(self.target_projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        features: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(features, tuple):
            source_features, target_features = features
            homogeneous = False
        else:
            source_features = target_features = features
            homogeneous = True
        source_projected = self.source_projection(source_features).view(
            -1, self.heads, self.output_dim
        )
        target_projected = self.target_projection(target_features).view(
            -1, self.heads, self.output_dim
        )
        if self.add_self_loops and homogeneous:
            nodes = torch.arange(
                len(target_features), device=edge_index.device
            )
            edge_index = torch.cat(
                [edge_index, torch.stack([nodes, nodes])], dim=1
            )
        source, target = edge_index
        source_score = (
            source_projected * self.attention_source
        ).sum(dim=-1)
        target_score = (
            target_projected * self.attention_target
        ).sum(dim=-1)
        score = F.leaky_relu(
            source_score[source] + target_score[target],
            negative_slope=self.negative_slope,
        )
        attention = segment_softmax(
            score, target, len(target_features)
        )
        messages = source_projected[source] * attention.unsqueeze(-1)
        output = segment_sum(messages, target, len(target_features))
        return output.reshape(
            len(target_features), self.heads * self.output_dim
        ) + self.bias


class SAGReadout(nn.Module):
    """SAGPooling(min_score=-1) followed by global_add_pool."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.neighbor = nn.Linear(feature_dim, 1, bias=False)
        self.root = nn.Linear(feature_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.neighbor.weight)
        nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.root.bias)

    def forward(self, data: MoleculeBatch) -> torch.Tensor:
        source, target = data.edge_index
        neighbor_messages = self.neighbor(data.x[source])
        neighbor_sum = segment_sum(
            neighbor_messages, target, len(data.x)
        )
        score = neighbor_sum + self.root(data.x)
        score = segment_softmax(
            score, data.batch, data.graph_count
        )
        return segment_sum(
            data.x * score, data.batch, data.graph_count
        )


@dataclass
class GraphPairBatch:
    head: MoleculeBatch
    tail: MoleculeBatch
    target: torch.Tensor
    indices: torch.Tensor
    bipartite_edge_index: torch.Tensor | None = None

    def to(self, device: torch.device) -> GraphPairBatch:
        return GraphPairBatch(
            head=self.head.to(device),
            tail=self.tail.to(device),
            target=self.target.to(device, non_blocking=True),
            indices=self.indices,
            bipartite_edge_index=(
                None
                if self.bipartite_edge_index is None
                else self.bipartite_edge_index.to(
                    device, non_blocking=True
                )
            ),
        )


class GraphPairCollator:
    def __init__(
        self,
        dataset: GraphPairDataset,
        graphs: dict[str, dict[str, torch.Tensor]],
        include_bipartite: bool,
    ) -> None:
        self.dataset = dataset
        self.graphs = graphs
        self.include_bipartite = include_bipartite
        missing = sorted(
            {
                drug_id
                for example in dataset.examples
                for drug_id in (example.d1, example.d2)
                if drug_id not in graphs
            }
        )
        if missing:
            raise ValueError(
                f"{len(missing)} drugs are absent from graph cache: "
                + ", ".join(missing[:10])
            )

    def __call__(self, indices: list[int]) -> GraphPairBatch:
        head_data = []
        tail_data = []
        head_offsets = []
        tail_offsets = []
        head_offset = 0
        tail_offset = 0
        for index in indices:
            example = self.dataset.examples[index]
            head_graph = self.graphs[example.d1]
            tail_graph = self.graphs[example.d2]
            head_data.append(head_graph)
            tail_data.append(tail_graph)
            head_offsets.append((head_offset, len(head_graph["x"])))
            tail_offsets.append((tail_offset, len(tail_graph["x"])))
            head_offset += len(head_graph["x"])
            tail_offset += len(tail_graph["x"])

        bipartite = None
        if self.include_bipartite:
            source_parts = []
            target_parts = []
            for (h_offset, h_count), (t_offset, t_count) in zip(
                head_offsets, tail_offsets
            ):
                source = torch.arange(h_count, dtype=torch.long) + h_offset
                target = torch.arange(t_count, dtype=torch.long) + t_offset
                source_grid, target_grid = torch.meshgrid(
                    source, target, indexing="ij"
                )
                source_parts.append(source_grid.reshape(-1))
                target_parts.append(target_grid.reshape(-1))
            bipartite = torch.stack(
                [torch.cat(source_parts), torch.cat(target_parts)]
            )
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return GraphPairBatch(
            head=batch_molecular_graphs(head_data),
            tail=batch_molecular_graphs(tail_data),
            target=self.dataset.targets[index_tensor],
            indices=index_tensor,
            bipartite_edge_index=bipartite,
        )


class CoAttentionLayer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        attention_dim = feature_dim // 2
        self.w_q = nn.Parameter(torch.empty(feature_dim, attention_dim))
        self.w_k = nn.Parameter(torch.empty(feature_dim, attention_dim))
        self.bias = nn.Parameter(torch.empty(attention_dim))
        self.a = nn.Parameter(torch.empty(attention_dim))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(-1, 1))
        nn.init.xavier_uniform_(self.a.view(-1, 1))

    def forward(
        self, receiver: torch.Tensor, attendant: torch.Tensor
    ) -> torch.Tensor:
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        activations = (
            queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        )
        return torch.tanh(activations) @ self.a


class MultiRelationRESCAL(nn.Module):
    def __init__(self, num_relations: int, feature_dim: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.feature_dim = feature_dim
        self.relation = nn.Parameter(
            torch.empty(num_relations, feature_dim * feature_dim)
        )
        nn.init.xavier_uniform_(self.relation)

    def forward(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        attention: torch.Tensor | None,
    ) -> torch.Tensor:
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        relations = F.normalize(self.relation, dim=-1).view(
            self.num_relations, self.feature_dim, self.feature_dim
        )
        scores = torch.einsum(
            "bkd,cde,ble->bckl", heads, relations, tails
        )
        if attention is not None:
            scores = scores * attention.unsqueeze(1)
        return scores.sum(dim=(-2, -1))


class SSIDDIBlock(nn.Module):
    def __init__(
        self, input_dim: int, head_output_dim: int, heads: int
    ) -> None:
        super().__init__()
        output_dim = head_output_dim * heads
        self.convolution = GraphAttentionConv(
            input_dim, head_output_dim, heads
        )
        self.readout = SAGReadout(output_dim)

    def forward(
        self, data: MoleculeBatch
    ) -> tuple[MoleculeBatch, torch.Tensor]:
        data.x = self.convolution(data.x, data.edge_index)
        return data, self.readout(data)


class SSIDDI(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = ATOM_FEATURE_DIM,
        kge_dim: int = 64,
        head_output_features: list[int] | tuple[int, ...] = (
            32,
            32,
            32,
            32,
        ),
        heads: list[int] | tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(head_output_features) != len(heads):
            raise ValueError("head_output_features and heads must match")
        self.initial_norm = GraphLayerNorm(input_dim)
        self.blocks = nn.ModuleList()
        self.net_norms = nn.ModuleList()
        current_dim = input_dim
        for head_output_dim, num_heads in zip(
            head_output_features, heads
        ):
            output_dim = int(head_output_dim) * int(num_heads)
            self.blocks.append(
                SSIDDIBlock(current_dim, head_output_dim, num_heads)
            )
            self.net_norms.append(GraphLayerNorm(output_dim))
            current_dim = output_dim
        if current_dim != kge_dim:
            raise ValueError(
                f"Final SSI block dimension {current_dim} != kge_dim {kge_dim}"
            )
        self.co_attention = CoAttentionLayer(kge_dim)
        self.decoder = MultiRelationRESCAL(num_classes, kge_dim)

    def forward(
        self,
        head: MoleculeBatch,
        tail: MoleculeBatch,
        bipartite_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bipartite_edge_index
        head.x = self.initial_norm(
            head.x, head.batch, head.graph_count
        )
        tail.x = self.initial_norm(
            tail.x, tail.batch, tail.graph_count
        )
        head_representations = []
        tail_representations = []
        for block, normalization in zip(self.blocks, self.net_norms):
            head, head_global = block(head)
            tail, tail_global = block(tail)
            head_representations.append(head_global)
            tail_representations.append(tail_global)
            head.x = F.elu(
                normalization(head.x, head.batch, head.graph_count)
            )
            tail.x = F.elu(
                normalization(tail.x, tail.batch, tail.graph_count)
            )
        heads = torch.stack(head_representations, dim=-2)
        tails = torch.stack(tail_representations, dim=-2)
        attention = self.co_attention(heads, tails)
        return self.decoder(heads, tails, attention)


class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.convolution = GraphAttentionConv(input_dim, 32, 2)

    def forward(self, data: MoleculeBatch) -> torch.Tensor:
        return self.convolution(F.elu(data.x), data.edge_index)


class InterGraphAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.convolution = GraphAttentionConv(
            (input_dim, input_dim),
            32,
            2,
            add_self_loops=False,
        )

    def forward(
        self,
        head: MoleculeBatch,
        tail: MoleculeBatch,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_input = F.elu(head.x)
        tail_input = F.elu(tail.x)
        tail_representation = self.convolution(
            (head_input, tail_input), edge_index
        )
        head_representation = self.convolution(
            (tail_input, head_input), edge_index[[1, 0]]
        )
        return head_representation, tail_representation


class DSNDDIBlock(nn.Module):
    def __init__(
        self, input_dim: int, head_output_dim: int, heads: int
    ) -> None:
        super().__init__()
        feature_dim = head_output_dim * heads
        self.feature_convolution = GraphAttentionConv(
            input_dim, head_output_dim, heads
        )
        self.intra_attention = IntraGraphAttention(feature_dim)
        self.inter_attention = InterGraphAttention(feature_dim)
        # Each intra/inter branch returns 64 features, giving 128 together.
        self.output_dim = 128
        self.readout = SAGReadout(self.output_dim)

    def forward(
        self,
        head: MoleculeBatch,
        tail: MoleculeBatch,
        bipartite_edge_index: torch.Tensor,
    ) -> tuple[
        MoleculeBatch,
        MoleculeBatch,
        torch.Tensor,
        torch.Tensor,
    ]:
        head.x = self.feature_convolution(head.x, head.edge_index)
        tail.x = self.feature_convolution(tail.x, tail.edge_index)
        head_intra = self.intra_attention(head)
        tail_intra = self.intra_attention(tail)
        head_inter, tail_inter = self.inter_attention(
            head, tail, bipartite_edge_index
        )
        head.x = torch.cat([head_intra, head_inter], dim=1)
        tail.x = torch.cat([tail_intra, tail_inter], dim=1)
        return (
            head,
            tail,
            self.readout(head),
            self.readout(tail),
        )


class DSNDDI(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = ATOM_FEATURE_DIM,
        kge_dim: int = 128,
        head_output_features: list[int] | tuple[int, ...] = (
            64,
            64,
            64,
            64,
        ),
        heads: list[int] | tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(head_output_features) != len(heads):
            raise ValueError("head_output_features and heads must match")
        if kge_dim != 128:
            raise ValueError("The released DSN-DDI block output requires 128")
        self.initial_norm = GraphLayerNorm(input_dim)
        self.blocks = nn.ModuleList()
        self.net_norms = nn.ModuleList()
        current_dim = input_dim
        for head_output_dim, num_heads in zip(
            head_output_features, heads
        ):
            block = DSNDDIBlock(
                current_dim, head_output_dim, num_heads
            )
            self.blocks.append(block)
            self.net_norms.append(GraphLayerNorm(block.output_dim))
            current_dim = block.output_dim
        self.co_attention = CoAttentionLayer(kge_dim)
        self.decoder = MultiRelationRESCAL(num_classes, kge_dim)

    def forward(
        self,
        head: MoleculeBatch,
        tail: MoleculeBatch,
        bipartite_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bipartite_edge_index is None:
            raise ValueError("DSN-DDI requires a bipartite atom graph")
        head.x = self.initial_norm(
            head.x, head.batch, head.graph_count
        )
        tail.x = self.initial_norm(
            tail.x, tail.batch, tail.graph_count
        )
        head_representations = []
        tail_representations = []
        for block, normalization in zip(self.blocks, self.net_norms):
            head, tail, head_global, tail_global = block(
                head, tail, bipartite_edge_index
            )
            head_representations.append(head_global)
            tail_representations.append(tail_global)
            head.x = F.elu(
                normalization(head.x, head.batch, head.graph_count)
            )
            tail.x = F.elu(
                normalization(tail.x, tail.batch, tail.graph_count)
            )
        heads = torch.stack(head_representations, dim=-2)
        tails = torch.stack(tail_representations, dim=-2)
        attention = self.co_attention(heads, tails)
        return self.decoder(heads, tails, attention)


def make_model(config: dict, num_classes: int) -> nn.Module:
    baseline = str(config["baseline"]).lower()
    model_config = dict(config.get("model", {}))
    if baseline == "ssi_ddi":
        return SSIDDI(num_classes=num_classes, **model_config)
    if baseline == "dsn_ddi":
        return DSNDDI(num_classes=num_classes, **model_config)
    raise ValueError(f"Unsupported graph baseline: {baseline}")


def build_datasets(
    config: dict,
    splits: tuple[str, ...],
) -> tuple[dict[str, GraphPairDataset], dict, dict]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    if audit["task_mode"] != "multilabel":
        raise ValueError("Graph baselines require task_mode=multilabel")
    num_classes = int(audit["type_count"])
    graphs = load_graph_cache(paths["graph_cache"])
    datasets = {
        split: GraphPairDataset(
            load_pair_examples(
                paths["records"],
                paths["split"],
                split,
                num_classes,
            ),
            num_classes,
        )
        for split in splits
    }
    return datasets, audit, graphs


def make_loader(
    dataset: GraphPairDataset,
    graphs: dict[str, dict[str, torch.Tensor]],
    config: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    training = config["training"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    baseline = str(config["baseline"]).lower()
    num_workers = int(training.get("num_workers", 0))
    options = {
        "dataset": dataset,
        "batch_size": int(training.get("batch_size", 256)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": generator,
        "collate_fn": GraphPairCollator(
            dataset,
            graphs,
            include_bipartite=baseline == "dsn_ddi",
        ),
    }
    if num_workers > 0:
        options["persistent_workers"] = True
    return DataLoader(**options)


def positive_class_weights(
    dataset: GraphPairDataset, maximum: float | None
) -> torch.Tensor:
    positives = dataset.targets.sum(dim=0, dtype=torch.float64)
    negatives = len(dataset) - positives
    weights = (negatives / positives.clamp_min(1)).float()
    return weights.clamp_max(maximum) if maximum is not None else weights


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    indices = []
    with torch.inference_mode():
        for batch in tqdm(
            loader, desc=description, leave=False, dynamic_ncols=True
        ):
            batch = batch.to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(
                    batch.head,
                    batch.tail,
                    batch.bipartite_edge_index,
                )
            probabilities.append(torch.sigmoid(logits.float()).cpu())
            targets.append(batch.target.cpu())
            indices.append(batch.indices)
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(indices).numpy(),
    )


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "learning_rate": float(row["learning_rate"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
            for row in csv.DictReader(handle)
        ]


def train(
    config: dict,
    resume_checkpoint: Path | None = None,
) -> None:
    assert_cloud_training_authorized()
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit, graphs = build_datasets(config, ("train", "val"))
    train_set = datasets["train"]
    val_set = datasets["val"]
    train_loader = make_loader(train_set, graphs, config, True, seed)
    val_loader = make_loader(val_set, graphs, config, False, seed + 1)
    training = config["training"]
    precision = str(training.get("mixed_precision", "none")).lower()
    if precision not in {"none", "bf16"}:
        raise ValueError("training.mixed_precision must be 'none' or 'bf16'")
    use_bf16 = precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but is unsupported on this GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_classes = int(audit["type_count"])
    model = make_model(config, num_classes)
    device = torch.device("cuda")
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-2)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training.get("lr_gamma", 0.96))
    )
    loss_config = config.get("loss", {})
    maximum_value = loss_config.get("max_class_weight")
    maximum = (
        float(maximum_value) if maximum_value is not None else None
    )
    weights = positive_class_weights(train_set, maximum).to(device)
    baseline = str(config["baseline"]).upper()
    print(
        f"Model: {baseline}, parameters={parameter_count:,}, "
        f"blocks={len(model.blocks)}"
    )
    print(
        f"Examples: train={len(train_set)}, val={len(val_set)}; "
        f"batch_size={training.get('batch_size')}, "
        f"mixed_precision={precision}"
    )
    print(
        "Class weights: "
        f"min={weights.min().item():.4f}, "
        f"max={weights.max().item():.4f}, cap={maximum}"
    )

    paths = {key: Path(value) for key, value in config["paths"].items()}
    checkpoint_dir = paths["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.csv"
    history: list[dict] = []
    best_f1 = -1.0
    stale = 0
    first_epoch = 0
    previous_elapsed = 0.0
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        if str(checkpoint.get("baseline", "")).lower() != str(
            config["baseline"]
        ).lower():
            raise ValueError("Resume checkpoint baseline does not match config")
        if int(checkpoint.get("num_classes", -1)) != num_classes:
            raise ValueError(
                "Resume checkpoint class count does not match audit"
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        first_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint.get("best_f1", -1.0))
        stale = int(checkpoint.get("stale", 0))
        previous_elapsed = float(
            checkpoint.get("elapsed_seconds", 0.0)
        )
        history = load_history(history_path)
        if history and int(history[-1]["epoch"]) != first_epoch:
            raise ValueError(
                "history.csv does not end at the resume checkpoint epoch"
            )
        print(
            f"Resumed from {resume_checkpoint}: completed_epoch={first_epoch}, "
            f"best_val_macro_f1={best_f1:.6f}, stale={stale}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
    max_epochs = int(training.get("epochs", 60))
    if first_epoch >= max_epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {first_epoch}, but "
            f"training.epochs={max_epochs}; increase the epoch limit."
        )
    patience = int(training.get("patience", 10))
    started = time.perf_counter()
    for epoch in range(first_epoch, max_epochs):
        model.train()
        running_loss = torch.zeros((), device=device)
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1:03d}/{max_epochs:03d}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(
                    batch.head,
                    batch.tail,
                    batch.bipartite_edge_index,
                )
                loss = classification_loss(
                    logits,
                    batch.target,
                    "multilabel",
                    weights,
                    loss_name=str(
                        loss_config.get("classification", "weighted")
                    ),
                    focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, step={step}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training.get("gradient_clip", 5.0)),
            )
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(
                    f"Non-finite gradient at epoch={epoch + 1}, step={step}"
                )
            optimizer.step()
            running_loss += loss.detach()
            if step == 1 or step % 20 == 0:
                progress.set_postfix(
                    loss=f"{(running_loss / step).item():.6f}",
                    refresh=False,
                )
        progress.close()
        epoch_loss = float(
            (running_loss / max(len(train_loader), 1)).item()
        )
        probability, target, _ = collect_predictions(
            model, val_loader, device, use_bf16, "validation"
        )
        val_f1 = macro_f1_at_half(target, probability)
        scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed = previous_elapsed + time.perf_counter() - started
        print(
            f"epoch={epoch + 1:03d} loss={epoch_loss:.6f} "
            f"val_macro_f1={val_f1:.6f} lr={learning_rate:.2e}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_macro_f1": val_f1,
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
            }
        )
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            stale = 0
        else:
            stale += 1
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1,
            "stale": stale,
            "elapsed_seconds": elapsed,
            "parameter_count": parameter_count,
            "num_classes": num_classes,
            "task_mode": "multilabel",
            "baseline": config["baseline"],
            "config": config,
            "validation": {
                "macro_f1": val_f1,
                "examples": len(val_set),
                "threshold": 0.5,
            },
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if improved:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        if stale >= patience:
            print("Early stopping.")
            break


def evaluate(
    config: dict,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires a CUDA GPU.")
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit, graphs = build_datasets(config, (split,))
    dataset = datasets[split]
    loader = make_loader(dataset, graphs, config, False, seed + 2)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    num_classes = int(audit["type_count"])
    if int(checkpoint["num_classes"]) != num_classes:
        raise ValueError("Checkpoint class count does not match audit")
    if str(checkpoint["baseline"]).lower() != str(
        config["baseline"]
    ).lower():
        raise ValueError("Checkpoint baseline does not match config")
    model = make_model(config, num_classes)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    precision = str(
        config["training"].get("mixed_precision", "none")
    ).lower()
    probability, target, indices = collect_predictions(
        model,
        loader,
        device,
        precision == "bf16",
        f"evaluate {split}",
    )
    prediction = (probability >= 0.5).astype(np.int8)
    metrics = {
        "macro_f1": macro_f1_at_half(target, probability),
        "examples": int(len(dataset)),
        "classes": num_classes,
        "threshold": 0.5,
        "dataset": audit.get("dataset"),
        "split": split,
        "task_mode": "multilabel",
        "baseline": config["baseline"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation": checkpoint.get("validation"),
        "parameter_count": checkpoint.get("parameter_count"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    sample_rows = []
    for sample_index, dataset_index in enumerate(indices.tolist()):
        example = dataset.examples[dataset_index]
        sample_rows.append(
            {
                "sample_index": sample_index,
                "d1": example.d1,
                "d2": example.d2,
                "record_ids": ";".join(
                    str(value) for value in example.record_ids
                ),
            }
        )
    with (output_dir / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=target,
        probability=probability,
        prediction=prediction,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train SSI-DDI or DSN-DDI on the fixed DCIA split."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--force", action="store_true")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Resume model, optimizer, scheduler, epoch and early-stopping "
            "state from latest.pt."
        ),
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument(
        "--split", choices=("val", "test"), default="test"
    )
    evaluate_parser.add_argument(
        "--output-dir", required=True, type=Path
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        prepare_graph_cache(config, force=args.force)
    elif args.command == "train":
        train(
            config,
            resume_checkpoint=(
                None
                if args.resume_checkpoint is None
                else args.resume_checkpoint.resolve()
            ),
        )
    else:
        evaluate(
            config,
            args.checkpoint.resolve(),
            args.split,
            args.output_dir.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
