from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .deepddi_baseline import (
    PairExample,
    file_sha256,
    load_pair_examples,
    macro_f1_at_half,
)
from .graph_ddi_baselines import (
    ATOM_FEATURE_DIM,
    CoAttentionLayer,
    GraphAttentionConv,
    GraphLayerNorm,
    GraphPairDataset,
    MoleculeBatch,
    MultiRelationRESCAL,
    batch_molecular_graphs,
    drug_ids_from_records,
    load_graph_cache,
    molecule_graph_from_smiles,
    positive_class_weights,
    segment_sum,
)
from .losses import classification_loss
from .train_cloud import (
    assert_cloud_training_authorized,
    load_audit,
    set_seed,
)


HIERARCHICAL_CACHE_VERSION = 1


def _connected_components_after_cuts(
    molecule: Chem.Mol, cut_bonds: set[int]
) -> list[list[int]]:
    adjacency = [[] for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        if bond.GetIdx() in cut_bonds:
            continue
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        adjacency[source].append(target)
        adjacency[target].append(source)
    components: list[list[int]] = []
    visited: set[int] = set()
    for start in range(molecule.GetNumAtoms()):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def hierarchical_graph_from_smiles(
    smiles: str,
) -> dict[str, torch.Tensor]:
    """Build the atom-fragment-molecule hierarchy used by HDN/HLN-DDI.

    BRICS bonds define chemically meaningful fragment nodes. Atom-fragment and
    fragment-molecule membership edges are bidirectional, while all original
    atom bonds are retained.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    atom_graph = molecule_graph_from_smiles(smiles)
    atom_count = molecule.GetNumAtoms()
    cut_bonds: set[int] = set()
    for (source, target), _labels in BRICS.FindBRICSBonds(molecule):
        bond = molecule.GetBondBetweenAtoms(int(source), int(target))
        if bond is not None:
            cut_bonds.add(bond.GetIdx())
    fragments = _connected_components_after_cuts(molecule, cut_bonds)
    if not fragments:
        fragments = [list(range(atom_count))]

    atom_x = atom_graph["x"]
    fragment_x = torch.stack(
        [atom_x[fragment].mean(dim=0) for fragment in fragments]
    )
    molecule_x = atom_x.mean(dim=0, keepdim=True)
    features = torch.cat([atom_x, fragment_x, molecule_x], dim=0)
    fragment_offset = atom_count
    molecule_index = atom_count + len(fragments)
    edge_parts = [atom_graph["edge_index"]]
    membership_edges: list[tuple[int, int]] = []
    for fragment_index, atoms in enumerate(fragments):
        node = fragment_offset + fragment_index
        for atom_index in atoms:
            membership_edges.extend(
                [(atom_index, node), (node, atom_index)]
            )
        membership_edges.extend(
            [(node, molecule_index), (molecule_index, node)]
        )
    edge_parts.append(
        torch.tensor(membership_edges, dtype=torch.long).t().contiguous()
    )
    node_type = torch.cat(
        [
            torch.zeros(atom_count, dtype=torch.long),
            torch.ones(len(fragments), dtype=torch.long),
            torch.full((1,), 2, dtype=torch.long),
        ]
    )
    return {
        "x": features,
        "edge_index": torch.cat(edge_parts, dim=1),
        "node_type": node_type,
    }


def prepare_hierarchical_cache(config: dict, force: bool = False) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    cache_path = paths["hierarchical_cache"]
    expected = set(drug_ids_from_records(paths["records"]))
    if cache_path.is_file() and not force:
        graphs = load_hierarchical_cache(cache_path)
        if expected.issubset(graphs):
            print(
                f"Hierarchical cache already covers all {len(expected)} "
                f"drugs: {cache_path}"
            )
            return
        raise ValueError(
            f"Existing cache is incomplete: {cache_path}; use --force"
        )

    graphs: dict[str, dict[str, torch.Tensor]] = {}
    for drug_id in tqdm(
        sorted(expected),
        desc="prepare BRICS hierarchical graphs",
        dynamic_ncols=True,
    ):
        archive_path = paths["conformers"] / f"{drug_id}.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Missing conformer archive for {drug_id}: {archive_path}"
            )
        with np.load(archive_path, allow_pickle=False) as archive:
            smiles = str(archive["canonical_smiles"].item())
        graphs[drug_id] = hierarchical_graph_from_smiles(smiles)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "version": HIERARCHICAL_CACHE_VERSION,
            "feature_dim": ATOM_FEATURE_DIM,
            "fragmentation": "rdkit_brics",
            "graphs": graphs,
        },
        temporary,
    )
    temporary.replace(cache_path)
    print(f"Saved {len(graphs)} hierarchical graphs to {cache_path}")


def load_hierarchical_cache(
    path: Path,
) -> dict[str, dict[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing hierarchical graph cache: {path}. Run prepare first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != HIERARCHICAL_CACHE_VERSION:
        raise ValueError(f"Unsupported hierarchical cache version in {path}")
    if int(payload.get("feature_dim", -1)) != ATOM_FEATURE_DIM:
        raise ValueError(f"Unexpected feature dimension in {path}")
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError(f"No hierarchical graphs in {path}")
    for drug_id, graph in graphs.items():
        node_type = graph.get("node_type")
        if node_type is None or set(node_type.tolist()) != {0, 1, 2}:
            raise ValueError(f"Invalid hierarchy for drug {drug_id}")
    return graphs


@dataclass
class HierarchicalBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    node_type: torch.Tensor
    graph_count: int

    def to(self, device: torch.device) -> HierarchicalBatch:
        return HierarchicalBatch(
            x=self.x.to(device, non_blocking=True),
            edge_index=self.edge_index.to(device, non_blocking=True),
            batch=self.batch.to(device, non_blocking=True),
            node_type=self.node_type.to(device, non_blocking=True),
            graph_count=self.graph_count,
        )


def batch_hierarchical_graphs(
    graphs: list[dict[str, torch.Tensor]],
) -> HierarchicalBatch:
    ordinary = batch_molecular_graphs(graphs)
    return HierarchicalBatch(
        x=ordinary.x,
        edge_index=ordinary.edge_index,
        batch=ordinary.batch,
        node_type=torch.cat([graph["node_type"] for graph in graphs]),
        graph_count=ordinary.graph_count,
    )


@dataclass
class HierarchicalPairBatch:
    head: HierarchicalBatch
    tail: HierarchicalBatch
    target: torch.Tensor
    indices: torch.Tensor
    fragment_edge_index: torch.Tensor

    def to(self, device: torch.device) -> HierarchicalPairBatch:
        return HierarchicalPairBatch(
            head=self.head.to(device),
            tail=self.tail.to(device),
            target=self.target.to(device, non_blocking=True),
            indices=self.indices,
            fragment_edge_index=self.fragment_edge_index.to(
                device, non_blocking=True
            ),
        )


class HierarchicalPairCollator:
    def __init__(
        self,
        dataset: GraphPairDataset,
        graphs: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        self.dataset = dataset
        self.graphs = graphs

    def __call__(self, indices: list[int]) -> HierarchicalPairBatch:
        head_graphs = []
        tail_graphs = []
        source_parts = []
        target_parts = []
        head_offset = 0
        tail_offset = 0
        for index in indices:
            example = self.dataset.examples[index]
            head = self.graphs[example.d1]
            tail = self.graphs[example.d2]
            head_graphs.append(head)
            tail_graphs.append(tail)
            head_fragments = (
                torch.nonzero(head["node_type"] == 1).flatten() + head_offset
            )
            tail_fragments = (
                torch.nonzero(tail["node_type"] == 1).flatten() + tail_offset
            )
            source_grid, target_grid = torch.meshgrid(
                head_fragments, tail_fragments, indexing="ij"
            )
            source_parts.append(source_grid.reshape(-1))
            target_parts.append(target_grid.reshape(-1))
            head_offset += len(head["x"])
            tail_offset += len(tail["x"])
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return HierarchicalPairBatch(
            head=batch_hierarchical_graphs(head_graphs),
            tail=batch_hierarchical_graphs(tail_graphs),
            target=self.dataset.targets[index_tensor],
            indices=index_tensor,
            fragment_edge_index=torch.stack(
                [torch.cat(source_parts), torch.cat(target_parts)]
            ),
        )


def _level_pool(data: HierarchicalBatch, level: int) -> torch.Tensor:
    mask = data.node_type == level
    pooled = segment_sum(
        data.x[mask], data.batch[mask], data.graph_count
    )
    count = torch.bincount(
        data.batch[mask], minlength=data.graph_count
    ).to(data.x.dtype)
    return pooled / count.clamp_min(1).unsqueeze(-1)


class HDNBlock(nn.Module):
    """Hierarchical-view plus fragment-level interactive-view block."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        per_head = hidden_dim // heads
        self.intra = GraphAttentionConv(hidden_dim, per_head, heads)
        self.inter = GraphAttentionConv(
            (hidden_dim, hidden_dim),
            per_head,
            heads,
            add_self_loops=False,
        )
        self.merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hierarchy = GraphAttentionConv(hidden_dim, per_head, heads)
        self.head_norm = GraphLayerNorm(hidden_dim)
        self.tail_norm = GraphLayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        head: HierarchicalBatch,
        tail: HierarchicalBatch,
        fragment_edge_index: torch.Tensor,
    ) -> tuple[HierarchicalBatch, HierarchicalBatch]:
        head_intra = self.intra(F.elu(head.x), head.edge_index)
        tail_intra = self.intra(F.elu(tail.x), tail.edge_index)
        tail_inter = self.inter(
            (F.elu(head.x), F.elu(tail.x)), fragment_edge_index
        )
        head_inter = self.inter(
            (F.elu(tail.x), F.elu(head.x)),
            fragment_edge_index[[1, 0]],
        )
        head_merged = self.merge(torch.cat([head_intra, head_inter], dim=-1))
        tail_merged = self.merge(torch.cat([tail_intra, tail_inter], dim=-1))
        head.x = self.hierarchy(head_merged, head.edge_index) + head.x
        tail.x = self.hierarchy(tail_merged, tail.edge_index) + tail.x
        head.x = F.dropout(
            F.elu(
                self.head_norm(head.x, head.batch, head.graph_count)
            ),
            self.dropout,
            self.training,
        )
        tail.x = F.dropout(
            F.elu(
                self.tail_norm(tail.x, tail.batch, tail.graph_count)
            ),
            self.dropout,
            self.training,
        )
        return head, tail


class HDNDDI(nn.Module):
    """HDN-DDI adapted to emit all labels for a directed drug pair."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        blocks: int = 6,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.initial_norm = GraphLayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [HDNBlock(hidden_dim, heads, dropout) for _ in range(blocks)]
        )
        self.co_attention = CoAttentionLayer(hidden_dim)
        self.decoder = MultiRelationRESCAL(num_classes, hidden_dim)

    def _initialize(self, data: HierarchicalBatch) -> None:
        data.x = (
            self.input_projection(data.x)
            + self.type_embedding(data.node_type)
        )
        data.x = F.elu(
            self.initial_norm(data.x, data.batch, data.graph_count)
        )

    def forward(self, batch: HierarchicalPairBatch) -> torch.Tensor:
        self._initialize(batch.head)
        self._initialize(batch.tail)
        head_layers = []
        tail_layers = []
        for block in self.blocks:
            batch.head, batch.tail = block(
                batch.head, batch.tail, batch.fragment_edge_index
            )
            head_layers.append(_level_pool(batch.head, 2))
            tail_layers.append(_level_pool(batch.tail, 2))
        heads = torch.stack(head_layers, dim=1)
        tails = torch.stack(tail_layers, dim=1)
        attention = self.co_attention(heads, tails)
        return self.decoder(heads, tails, attention)


class HLNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.convolutions = nn.ModuleList(
            [
                GraphAttentionConv(
                    hidden_dim, hidden_dim // heads, heads
                )
                for _ in range(layers)
            ]
        )
        self.normalizations = nn.ModuleList(
            [GraphLayerNorm(hidden_dim) for _ in range(layers)]
        )
        self.dropout = dropout

    def forward(self, data: HierarchicalBatch) -> torch.Tensor:
        data.x = (
            self.input_projection(data.x)
            + self.type_embedding(data.node_type)
        )
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            residual = data.x
            data.x = convolution(F.elu(data.x), data.edge_index) + residual
            data.x = F.dropout(
                F.elu(
                    normalization(data.x, data.batch, data.graph_count)
                ),
                self.dropout,
                self.training,
            )
        return torch.stack(
            [_level_pool(data, level) for level in (0, 1, 2)], dim=1
        )


class HLNDDI(nn.Module):
    """Atom/motif/super-node hierarchy with co-attention and RESCAL."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        layers: int = 5,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = HLNEncoder(
            input_dim, hidden_dim, layers, heads, dropout
        )
        self.co_attention = CoAttentionLayer(hidden_dim)
        self.decoder = MultiRelationRESCAL(num_classes, hidden_dim)
        self.blocks = self.encoder.convolutions

    def forward(self, batch: HierarchicalPairBatch) -> torch.Tensor:
        heads = self.encoder(batch.head)
        tails = self.encoder(batch.tail)
        attention = self.co_attention(heads, tails)
        return self.decoder(heads, tails, attention)


class RelationalGraphConv(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_relations: int
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_relations, input_dim, output_dim)
        )
        self.root = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.root.weight)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        output = features.new_zeros((len(features), self.weight.shape[-1]))
        # Grouping edges by relation avoids materializing an
        # [edge, input_dim, output_dim] tensor. On the full Ryu graph that
        # naive expansion is several gigabytes even though the actual RGCN
        # state is small.
        for relation in torch.unique(edge_type):
            mask = edge_type == relation
            relation_source = source[mask]
            relation_target = target[mask]
            messages = features[relation_source] @ self.weight[relation]
            output.index_add_(0, relation_target, messages)
        degree = torch.bincount(
            target, minlength=len(features)
        ).to(features.dtype)
        output = output / degree.clamp_min(1).unsqueeze(-1)
        return output + self.root(features) + self.bias


class DGIDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(
        self, summary: torch.Tensor, representation: torch.Tensor
    ) -> torch.Tensor:
        return self.score(
            representation, summary.expand_as(representation)
        ).squeeze(-1)


@dataclass
class MRCPairBatch:
    head_index: torch.Tensor
    tail_index: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor

    def to(self, device: torch.device) -> MRCPairBatch:
        return MRCPairBatch(
            head_index=self.head_index.to(device, non_blocking=True),
            tail_index=self.tail_index.to(device, non_blocking=True),
            target=self.target.to(device, non_blocking=True),
            indices=self.indices,
        )


class MRCPairCollator:
    def __init__(
        self,
        dataset: GraphPairDataset,
        drug_to_index: dict[str, int],
    ) -> None:
        self.dataset = dataset
        self.drug_to_index = drug_to_index

    def __call__(self, indices: list[int]) -> MRCPairBatch:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        examples = [self.dataset.examples[index] for index in indices]
        return MRCPairBatch(
            head_index=torch.tensor(
                [self.drug_to_index[item.d1] for item in examples],
                dtype=torch.long,
            ),
            tail_index=torch.tensor(
                [self.drug_to_index[item.d2] for item in examples],
                dtype=torch.long,
            ),
            target=self.dataset.targets[index_tensor],
            indices=index_tensor,
        )


def build_mrc_context(
    records_path: Path,
    split_path: Path,
    graph_cache_path: Path,
    num_classes: int,
) -> dict:
    """Create a leakage-safe relation graph from training pairs only."""
    graphs = load_graph_cache(graph_cache_path)
    drug_ids = drug_ids_from_records(records_path)
    missing = sorted(set(drug_ids) - set(graphs))
    if missing:
        raise ValueError(f"MRCGNN graph cache misses {len(missing)} drugs")
    drug_to_index = {
        drug_id: index for index, drug_id in enumerate(drug_ids)
    }
    initial_features = torch.stack(
        [graphs[drug_id]["x"].mean(dim=0) for drug_id in drug_ids]
    )
    train_examples = load_pair_examples(
        records_path, split_path, "train", num_classes
    )
    edges = []
    relation_types = []
    for example in train_examples:
        head = drug_to_index[example.d1]
        tail = drug_to_index[example.d2]
        for relation in example.labels:
            edges.extend([(head, tail), (tail, head)])
            relation_types.extend([relation, relation])
    if not edges:
        raise ValueError("The MRCGNN training relation graph has no edges")
    return {
        "drug_ids": drug_ids,
        "drug_to_index": drug_to_index,
        "initial_features": initial_features,
        "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
        "edge_type": torch.tensor(relation_types, dtype=torch.long),
        "training_pairs": len(train_examples),
    }


class MRCGNN(nn.Module):
    """Multi-relational graph contrastive network using train edges only."""

    def __init__(
        self,
        num_classes: int,
        context: dict,
        input_dim: int = ATOM_FEATURE_DIM,
        structural_dim: int = 128,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.5,
        decoder_hidden: int = 256,
    ) -> None:
        super().__init__()
        initial = context["initial_features"]
        if initial.shape[1] != input_dim:
            raise ValueError("MRCGNN initial feature dimension mismatch")
        self.register_buffer("node_features", initial)
        self.register_buffer("edge_index", context["edge_index"])
        self.register_buffer("edge_type", context["edge_type"])
        # The released MRCGNN consumes 128-dimensional pretrained TrimNet
        # features. This trainable projection keeps that representation width
        # while allowing the unified pipeline to start from its audited
        # 55-dimensional molecular descriptors.
        self.structure_encoder = nn.Sequential(
            nn.Linear(input_dim, structural_dim),
            nn.ELU(),
            nn.LayerNorm(structural_dim),
        )
        self.encoder1 = RelationalGraphConv(
            structural_dim, hidden1, num_classes
        )
        self.encoder2 = RelationalGraphConv(
            hidden1, hidden2, num_classes
        )
        self.layer_attention = nn.Parameter(torch.full((2,), 0.5))
        self.discriminator = DGIDiscriminator(hidden2)
        pair_dim = 2 * (hidden1 + hidden2 + structural_dim)
        self.decoder = nn.Sequential(
            nn.Linear(pair_dim, decoder_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden // 2, num_classes),
        )
        self.dropout = dropout
        self.blocks = nn.ModuleList([self.encoder1, self.encoder2])

    def _encode(
        self, features: torch.Tensor, edge_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first = F.relu(
            self.encoder1(features, self.edge_index, edge_type)
        )
        first = F.dropout(first, self.dropout, self.training)
        second = self.encoder2(first, self.edge_index, edge_type)
        return first, second

    def _contrastive(
        self, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        summary = torch.sigmoid(positive.mean(dim=0, keepdim=True))
        positive_score = self.discriminator(summary, positive)
        negative_score = self.discriminator(summary, negative)
        return F.binary_cross_entropy_with_logits(
            positive_score, torch.ones_like(positive_score)
        ) + F.binary_cross_entropy_with_logits(
            negative_score, torch.zeros_like(negative_score)
        )

    def forward(
        self, batch: MRCPairBatch, compute_auxiliary: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structural = self.structure_encoder(self.node_features)
        first, second = self._encode(structural, self.edge_type)
        weighted = torch.cat(
            [
                self.layer_attention[0] * first,
                self.layer_attention[1] * second,
                structural,
            ],
            dim=-1,
        )
        heads = weighted[batch.head_index]
        tails = weighted[batch.tail_index]
        logits = self.decoder(torch.cat([heads, tails], dim=-1))
        auxiliary: dict[str, torch.Tensor] = {}
        if compute_auxiliary:
            node_permutation = torch.randperm(
                len(self.node_features), device=self.node_features.device
            )
            _, corrupted_nodes = self._encode(
                structural[node_permutation], self.edge_type
            )
            relation_permutation = torch.randperm(
                len(self.edge_type), device=self.edge_type.device
            )
            _, corrupted_relations = self._encode(
                structural, self.edge_type[relation_permutation]
            )
            auxiliary["node_contrastive"] = self._contrastive(
                second, corrupted_nodes
            )
            auxiliary["relation_contrastive"] = self._contrastive(
                second, corrupted_relations
            )
        return logits, auxiliary


def _load_datasets(
    config: dict, splits: tuple[str, ...]
) -> tuple[dict[str, GraphPairDataset], dict]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    if audit["task_mode"] != "multilabel":
        raise ValueError("Recent DDI baselines require multilabel data")
    num_classes = int(audit["type_count"])
    datasets = {
        split: GraphPairDataset(
            load_pair_examples(
                paths["records"], paths["split"], split, num_classes
            ),
            num_classes,
        )
        for split in splits
    }
    return datasets, audit


def _build_runtime(config: dict, splits: tuple[str, ...]):
    paths = {key: Path(value) for key, value in config["paths"].items()}
    datasets, audit = _load_datasets(config, splits)
    baseline = str(config["baseline"]).lower()
    num_classes = int(audit["type_count"])
    context = None
    if baseline == "mrcgnn":
        context = build_mrc_context(
            paths["records"],
            paths["split"],
            paths["graph_cache"],
            num_classes,
        )
        model = MRCGNN(
            num_classes=num_classes,
            context=context,
            **dict(config.get("model", {})),
        )
    else:
        graphs = load_hierarchical_cache(paths["hierarchical_cache"])
        if baseline == "hdn_ddi":
            model = HDNDDI(
                num_classes=num_classes, **dict(config.get("model", {}))
            )
        elif baseline == "hln_ddi":
            model = HLNDDI(
                num_classes=num_classes, **dict(config.get("model", {}))
            )
        else:
            raise ValueError(f"Unsupported recent baseline: {baseline}")
        context = {"graphs": graphs}
    return datasets, audit, model, context


def _make_loader(
    dataset: GraphPairDataset,
    config: dict,
    context: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    baseline = str(config["baseline"]).lower()
    if baseline == "mrcgnn":
        collator = MRCPairCollator(dataset, context["drug_to_index"])
    else:
        collator = HierarchicalPairCollator(dataset, context["graphs"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(config["training"].get("num_workers", 0))
    options = {
        "dataset": dataset,
        "batch_size": int(config["training"].get("batch_size", 128)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "generator": generator,
        "collate_fn": collator,
    }
    if workers > 0:
        options["persistent_workers"] = True
    return DataLoader(**options)


def _model_forward(
    model: nn.Module,
    batch: MRCPairBatch | HierarchicalPairBatch,
    compute_auxiliary: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(model, MRCGNN):
        if not isinstance(batch, MRCPairBatch):
            raise TypeError("MRCGNN received an incompatible batch")
        return model(batch, compute_auxiliary)
    if not isinstance(batch, HierarchicalPairBatch):
        raise TypeError("Hierarchical model received an incompatible batch")
    return model(batch), {}


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
                logits, _ = _model_forward(model, batch, False)
            probabilities.append(torch.sigmoid(logits.float()).cpu())
            targets.append(batch.target.cpu())
            indices.append(batch.indices)
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(indices).numpy(),
    )


def _load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "classification_loss": float(row["classification_loss"]),
                "auxiliary_loss": float(row["auxiliary_loss"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "learning_rate": float(row["learning_rate"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
            for row in csv.DictReader(handle)
        ]


def train(config: dict, resume_checkpoint: Path | None = None) -> None:
    assert_cloud_training_authorized()
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit, model, context = _build_runtime(
        config, ("train", "val")
    )
    train_set = datasets["train"]
    val_set = datasets["val"]
    train_loader = _make_loader(train_set, config, context, True, seed)
    val_loader = _make_loader(val_set, config, context, False, seed + 1)
    training = config["training"]
    precision = str(training.get("mixed_precision", "none")).lower()
    if precision not in {"none", "bf16"}:
        raise ValueError("mixed_precision must be none or bf16")
    use_bf16 = precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training.get("lr_gamma", 0.96))
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    loss_config = config.get("loss", {})
    maximum_value = loss_config.get("max_class_weight")
    maximum = (
        None if maximum_value is None else float(maximum_value)
    )
    weights = positive_class_weights(train_set, maximum).to(device)
    print(
        f"Model: {str(config['baseline']).upper()}, "
        f"parameters={parameter_count:,}, blocks={len(model.blocks)}"
    )
    print(
        f"Examples: train={len(train_set)}, val={len(val_set)}; "
        f"batch_size={training.get('batch_size')}, "
        f"mixed_precision={precision}"
    )
    if str(config["baseline"]).lower() == "mrcgnn":
        print(
            "Leakage guard: MRCGNN relation graph contains only training "
            f"pairs ({context['training_pairs']} grouped pairs)."
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
            raise ValueError("Resume checkpoint baseline mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        first_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint.get("best_f1", -1.0))
        stale = int(checkpoint.get("stale", 0))
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        history = _load_history(history_path)
        print(
            f"Resumed from {resume_checkpoint}: epoch={first_epoch}, "
            f"best_val_macro_f1={best_f1:.6f}, stale={stale}"
        )

    max_epochs = int(training.get("epochs", 100))
    patience = int(training.get("patience", 15))
    if first_epoch >= max_epochs:
        raise ValueError("Resume epoch is not below training.epochs")
    node_ratio = float(loss_config.get("node_contrastive_ratio", 0.05))
    relation_ratio = float(
        loss_config.get("relation_contrastive_ratio", 0.1)
    )
    started = time.perf_counter()
    for epoch in range(first_epoch, max_epochs):
        model.train()
        running = torch.zeros(3, device=device)
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
                logits, auxiliary = _model_forward(model, batch, True)
                classification = classification_loss(
                    logits,
                    batch.target,
                    "multilabel",
                    weights,
                    loss_name=str(
                        loss_config.get("classification", "weighted")
                    ),
                    focal_gamma=float(
                        loss_config.get("focal_gamma", 2.0)
                    ),
                )
                auxiliary_loss = (
                    node_ratio
                    * auxiliary.get(
                        "node_contrastive",
                        classification.new_zeros(()),
                    )
                    + relation_ratio
                    * auxiliary.get(
                        "relation_contrastive",
                        classification.new_zeros(()),
                    )
                )
                loss = classification + auxiliary_loss
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
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            running += torch.stack(
                [loss.detach(), classification.detach(), auxiliary_loss.detach()]
            )
            if step == 1 or step % 20 == 0:
                progress.set_postfix(
                    loss=f"{(running[0] / step).item():.6f}",
                    refresh=False,
                )
        means = running / max(len(train_loader), 1)
        probability, target, _ = collect_predictions(
            model, val_loader, device, use_bf16, "validation"
        )
        val_f1 = macro_f1_at_half(target, probability)
        scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed = previous_elapsed + time.perf_counter() - started
        print(
            f"epoch={epoch + 1:03d} loss={means[0].item():.6f} "
            f"classification={means[1].item():.6f} "
            f"auxiliary={means[2].item():.6f} "
            f"val_macro_f1={val_f1:.6f} lr={learning_rate:.2e}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(means[0].item()),
                "classification_loss": float(means[1].item()),
                "auxiliary_loss": float(means[2].item()),
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
            "num_classes": int(audit["type_count"]),
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
        raise SystemExit("Evaluation requires a CUDA GPU")
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit, model, context = _build_runtime(config, (split,))
    dataset = datasets[split]
    loader = _make_loader(dataset, config, context, False, seed + 2)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if str(checkpoint["baseline"]).lower() != str(
        config["baseline"]
    ).lower():
        raise ValueError("Checkpoint baseline mismatch")
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
        "classes": int(audit["type_count"]),
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
    rows = []
    for sample_index, dataset_index in enumerate(indices.tolist()):
        example: PairExample = dataset.examples[dataset_index]
        rows.append(
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
        description="Train MRCGNN, HDN-DDI, or HLN-DDI on DCIA splits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--force", action="store_true")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume-checkpoint", type=Path)
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
    baseline = str(config["baseline"]).lower()
    if args.command == "prepare":
        if baseline == "mrcgnn":
            path = Path(config["paths"]["graph_cache"])
            load_graph_cache(path)
            print(f"MRCGNN molecular graph cache verified: {path}")
        else:
            prepare_hierarchical_cache(config, force=args.force)
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
