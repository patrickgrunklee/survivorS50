"""Graph Neural Network model for Survivor elimination prediction.

Models alliances and social dynamics as a dynamic graph where:
- Nodes = remaining players at each tribal council
- Edges = alliance (co-voting) and adversarial (cross-voting) relationships
- Node features = player stats relative to remaining field
- Prediction = which node (player) gets eliminated

Uses vote_history.csv from survivoR2py to build the social graph.
Each tribal council produces one graph snapshot for training.

Architecture: GAT (Graph Attention Network) with signed edges,
predicting per-node elimination probability via softmax.

Requires: pip install torch torch_geometric
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import SurvivorModel

# --------------------------------------------------------------------------- #
# Optional dependency gating (mirrors keras_models.py pattern)
# --------------------------------------------------------------------------- #
_PYG_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv
    _PYG_AVAILABLE = True
except ImportError:
    pass


def _check_pyg():
    if not _PYG_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric not installed. Run: pip install torch torch_geometric"
        )


# --------------------------------------------------------------------------- #
# Data paths
# --------------------------------------------------------------------------- #
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "survivoR2py"


# =========================================================================== #
#                        GRAPH CONSTRUCTION PIPELINE                          #
# =========================================================================== #

class TribalCouncilGraphBuilder:
    """Builds PyG graph snapshots from survivoR2py vote_history data.

    One graph per tribal council, where:
    - Nodes are the players eligible to vote at that tribal
    - Edge weights encode alliance strength (positive) or adversarial
      tension (negative) based on recent voting patterns
    - Node features capture per-player stats relative to remaining field

    Graph construction assumptions:
    - Alliance edge: two players voted for the SAME target at a tribal.
      Weight = number of recent co-votes, with exponential time decay.
    - Adversarial edge: player A voted for player B (or vice versa).
      Weight = negative, scaled by recency.
    - Edges are accumulated over a sliding window of recent tribals
      (default 5) to capture evolving alliance dynamics.
    - Nullified votes still signal intent — they're included in graph
      construction since they reveal alliance/adversarial relationships
      even when the vote doesn't count mechanically.
    - Self-votes and votes for players not at tribal are dropped.
    """

    # Return seasons used for primary training
    RETURN_SEASONS = [8, 20, 34, 40]
    # All available seasons — discovered from data at build time.
    # This class attr is a fallback; build_dataset reads the actual CSV.
    ALL_SEASONS = list(range(1, 51))

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        window_size: int = 5,
        decay_rate: float = 0.7,
    ):
        """
        Args:
            data_dir: Path to survivoR2py CSV directory.
            window_size: Number of recent tribals to consider for edge
                construction. Older tribals beyond this window are ignored.
            decay_rate: Exponential decay factor for edge weights.
                Weight from k tribals ago is multiplied by decay_rate^k.
                0.7 means a vote from 3 tribals ago has 0.7^3 = 0.34x weight.
        """
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.window_size = window_size
        self.decay_rate = decay_rate

        self._vote_history: Optional[pd.DataFrame] = None
        self._castaways: Optional[pd.DataFrame] = None
        self._challenge_results: Optional[pd.DataFrame] = None
        self._boot_mapping: Optional[pd.DataFrame] = None

    # ---- Lazy data loading ------------------------------------------------ #

    def _load_data(self):
        """Load and cache all required CSVs."""
        if self._vote_history is not None:
            return

        self._vote_history = pd.read_csv(self.data_dir / "vote_history.csv")
        self._castaways = pd.read_csv(self.data_dir / "castaways.csv")
        self._challenge_results = pd.read_csv(
            self.data_dir / "challenge_results.csv"
        )
        self._boot_mapping = pd.read_csv(self.data_dir / "boot_mapping.csv")

        # Normalize season/episode to int for consistent indexing
        for df in [self._vote_history, self._castaways,
                    self._challenge_results, self._boot_mapping]:
            for col in ["season", "episode"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
            if "order" in df.columns:
                df["order"] = df["order"].fillna(0).astype(int)

    # ---- Per-season tribal ordering --------------------------------------- #

    def _get_tribal_sequence(self, season: int) -> list[tuple[int, int, int]]:
        """Return ordered list of (season, episode, order) for all tribals.

        Each tuple uniquely identifies a tribal council within a season.
        Multiple tribals per episode are distinguished by `order`.
        """
        self._load_data()
        vh = self._vote_history
        season_votes = vh[vh["season"] == season]
        tribals = (
            season_votes
            .groupby(["season", "episode", "order"])
            .first()
            .reset_index()
            [["season", "episode", "order"]]
            .sort_values(["episode", "order"])
        )
        return list(tribals.itertuples(index=False, name=None))

    # ---- Players at a tribal ---------------------------------------------- #

    def _get_players_at_tribal(
        self, season: int, episode: int, order: int
    ) -> list[str]:
        """Return list of castaway names who voted at this tribal.

        Uses the actual vote records rather than boot_mapping to get the
        precise set of players present at this specific tribal council
        (handles split tribes, exile, etc.).
        """
        self._load_data()
        vh = self._vote_history
        tribal_votes = vh[
            (vh["season"] == season)
            & (vh["episode"] == episode)
            & (vh["order"] == order)
        ]
        # All voters at this tribal (drop NaN names)
        voters = set(
            c for c in tribal_votes["castaway"].unique() if pd.notna(c)
        )
        # Also include the voted_out person (they were present)
        voted_out = tribal_votes["voted_out"].iloc[0] if len(tribal_votes) > 0 else None
        if voted_out and pd.notna(voted_out):
            voters.add(voted_out)
        return sorted(voters)

    # ---- Edge construction ------------------------------------------------ #

    def _build_edges(
        self,
        season: int,
        tribal_idx: int,
        tribal_sequence: list[tuple[int, int, int]],
        players: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build alliance and adversarial edges from recent vote history.

        Returns:
            edge_index: (2, E) array of [source, target] node indices
            edge_weight: (E,) array of signed edge weights
            edge_type: (E,) array — 0 = alliance, 1 = adversarial
        """
        self._load_data()
        vh = self._vote_history
        player_to_idx = {p: i for i, p in enumerate(players)}
        player_set = set(players)

        # Sliding window: recent tribals up to (but not including) current
        start = max(0, tribal_idx - self.window_size)
        recent_tribals = tribal_sequence[start:tribal_idx]

        # Accumulate edge weights
        # Key: (i, j) sorted pair → {'alliance': float, 'adversarial': float}
        edge_acc: dict[tuple[int, int], dict[str, float]] = {}

        for k, (s, ep, ord_) in enumerate(recent_tribals):
            recency = len(recent_tribals) - k  # 1 = most recent
            decay = self.decay_rate ** (recency - 1)

            tribal_votes = vh[
                (vh["season"] == s)
                & (vh["episode"] == ep)
                & (vh["order"] == ord_)
            ]

            # --- Alliance edges: players who voted for the same target ---
            # Group voters by their vote target
            vote_groups: dict[str, list[str]] = {}
            for _, row in tribal_votes.iterrows():
                voter = row["castaway"]
                target = row["vote"]
                if pd.isna(target) or voter not in player_set:
                    continue
                vote_groups.setdefault(target, []).append(voter)

            for target, co_voters in vote_groups.items():
                # Every pair of co-voters gets an alliance edge
                for a_idx in range(len(co_voters)):
                    for b_idx in range(a_idx + 1, len(co_voters)):
                        a, b = co_voters[a_idx], co_voters[b_idx]
                        if a not in player_to_idx or b not in player_to_idx:
                            continue
                        i, j = player_to_idx[a], player_to_idx[b]
                        key = (min(i, j), max(i, j))
                        if key not in edge_acc:
                            edge_acc[key] = {"alliance": 0.0, "adversarial": 0.0}
                        edge_acc[key]["alliance"] += decay

            # --- Adversarial edges: player A voted for player B ---
            for _, row in tribal_votes.iterrows():
                voter = row["castaway"]
                target = row["vote"]
                if pd.isna(target):
                    continue
                if voter in player_to_idx and target in player_to_idx:
                    i, j = player_to_idx[voter], player_to_idx[target]
                    key = (min(i, j), max(i, j))
                    if key not in edge_acc:
                        edge_acc[key] = {"alliance": 0.0, "adversarial": 0.0}
                    edge_acc[key]["adversarial"] += decay

        # Convert to edge arrays (undirected: add both directions)
        src_list, dst_list, weight_list, type_list = [], [], [], []

        for (i, j), weights in edge_acc.items():
            # Net weight: alliance is positive, adversarial is negative
            alliance_w = weights["alliance"]
            adversarial_w = weights["adversarial"]

            if alliance_w > 0:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                weight_list.extend([alliance_w, alliance_w])
                type_list.extend([0, 0])

            if adversarial_w > 0:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                weight_list.extend([-adversarial_w, -adversarial_w])
                type_list.extend([1, 1])

        if len(src_list) == 0:
            # First tribal of the season — no prior history.
            # Add self-loops so GNN layers still propagate features.
            n = len(players)
            src_list = list(range(n))
            dst_list = list(range(n))
            weight_list = [0.0] * n
            type_list = [0] * n

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_weight = np.array(weight_list, dtype=np.float32)
        edge_type = np.array(type_list, dtype=np.int64)

        return edge_index, edge_weight, edge_type

    # ---- Node features ---------------------------------------------------- #

    def _build_node_features(
        self,
        season: int,
        episode: int,
        order: int,
        players: list[str],
        tribal_idx: int,
        tribal_sequence: list[tuple[int, int, int]],
    ) -> np.ndarray:
        """Build node feature matrix for players at this tribal.

        Features (per node, 10 dimensions):
          0: votes_received_recently — how many votes this player has
             received in the sliding window (normalized). Higher = more
             targeted = more vulnerable.
          1: votes_cast_for_eliminated — fraction of recent votes this
             player cast that matched the person actually eliminated.
             Higher = better at reading the room / in the majority.
          2: times_in_majority — fraction of recent tribals where this
             player voted with the majority.
          3: unique_allies — number of distinct co-voters in recent window
             (normalized by group size).
          4: unique_adversaries — number of distinct players who voted
             against this player recently (normalized).
          5: has_individual_immunity — 1.0 if player won individual
             immunity for this episode, 0.0 otherwise.
          6: is_immune — 1.0 if player has any form of immunity at this
             tribal (individual, hidden idol, etc.), 0.0 otherwise.
          7: tribal_position — fraction of the game elapsed.
             Early game = 0, late game = 1.
          8: degree_centrality — this player's edge count / max possible
             edges in the current graph window.
          9: vote_entropy — how spread out this player's recent votes
             are across targets. High entropy = no clear target = swing
             voter behavior.

        All features are normalized to [0, 1] range within each tribal
        to make them relative to the remaining field.
        """
        self._load_data()
        vh = self._vote_history
        n_players = len(players)
        features = np.zeros((n_players, 10), dtype=np.float32)
        player_to_idx = {p: i for i, p in enumerate(players)}
        player_set = set(players)

        # Sliding window of recent tribals (same as edge construction)
        start = max(0, tribal_idx - self.window_size)
        recent_tribals = tribal_sequence[start:tribal_idx]

        # --- Accumulate stats from recent tribals ---
        votes_received = np.zeros(n_players)
        votes_with_majority = np.zeros(n_players)
        tribals_participated = np.zeros(n_players)
        correct_votes = np.zeros(n_players)
        total_votes_cast = np.zeros(n_players)
        allies_sets = [set() for _ in range(n_players)]
        adversaries_sets = [set() for _ in range(n_players)]
        vote_target_counts: list[dict[str, int]] = [
            {} for _ in range(n_players)
        ]

        for s, ep, ord_ in recent_tribals:
            tribal_votes = vh[
                (vh["season"] == s)
                & (vh["episode"] == ep)
                & (vh["order"] == ord_)
            ]
            if len(tribal_votes) == 0:
                continue

            voted_out = tribal_votes["voted_out"].iloc[0]

            # Who voted for whom
            voter_target = {}
            for _, row in tribal_votes.iterrows():
                voter = row["castaway"]
                target = row["vote"]
                if pd.isna(target) or voter not in player_set:
                    continue
                voter_target[voter] = target

            # Majority vote target
            from collections import Counter
            target_counts = Counter(voter_target.values())
            majority_target = (
                target_counts.most_common(1)[0][0]
                if target_counts else None
            )

            # Group by target for co-voter detection
            target_groups: dict[str, list[str]] = {}
            for voter, target in voter_target.items():
                target_groups.setdefault(target, []).append(voter)

            for voter, target in voter_target.items():
                if voter not in player_to_idx:
                    continue
                idx = player_to_idx[voter]
                tribals_participated[idx] += 1
                total_votes_cast[idx] += 1

                # Track vote targets for entropy
                vote_target_counts[idx][target] = (
                    vote_target_counts[idx].get(target, 0) + 1
                )

                # Correct vote (voted for the person eliminated)
                if target == voted_out:
                    correct_votes[idx] += 1

                # Majority vote
                if target == majority_target:
                    votes_with_majority[idx] += 1

                # Co-voters = allies
                for co_voter in target_groups.get(target, []):
                    if co_voter != voter and co_voter in player_to_idx:
                        allies_sets[idx].add(co_voter)

            # Votes received
            for voter, target in voter_target.items():
                if target in player_to_idx:
                    adversaries_sets[player_to_idx[target]].add(voter)
                    votes_received[player_to_idx[target]] += 1

        # --- Feature 0: votes_received_recently ---
        features[:, 0] = votes_received

        # --- Feature 1: votes_cast_for_eliminated ---
        with np.errstate(divide="ignore", invalid="ignore"):
            features[:, 1] = np.where(
                total_votes_cast > 0,
                correct_votes / total_votes_cast,
                0.0,
            )

        # --- Feature 2: times_in_majority ---
        with np.errstate(divide="ignore", invalid="ignore"):
            features[:, 2] = np.where(
                tribals_participated > 0,
                votes_with_majority / tribals_participated,
                0.0,
            )

        # --- Feature 3: unique_allies (normalized) ---
        ally_counts = np.array([len(s) for s in allies_sets], dtype=np.float32)
        features[:, 3] = ally_counts / max(n_players - 1, 1)

        # --- Feature 4: unique_adversaries (normalized) ---
        adv_counts = np.array(
            [len(s) for s in adversaries_sets], dtype=np.float32
        )
        features[:, 4] = adv_counts / max(n_players - 1, 1)

        # --- Feature 5-6: immunity status ---
        tribal_votes = vh[
            (vh["season"] == season)
            & (vh["episode"] == episode)
            & (vh["order"] == order)
        ]
        for _, row in tribal_votes.iterrows():
            castaway = row["castaway"]
            if castaway in player_to_idx and pd.notna(row.get("immunity")):
                idx = player_to_idx[castaway]
                immunity_type = str(row["immunity"])
                if "Individual" in immunity_type:
                    features[idx, 5] = 1.0
                # Any form of immunity
                features[idx, 6] = 1.0

        # --- Feature 7: tribal_position (game progress) ---
        total_tribals = len(tribal_sequence)
        if total_tribals > 1:
            features[:, 7] = tribal_idx / (total_tribals - 1)

        # --- Feature 8: degree_centrality ---
        # Use ally + adversary connections as proxy
        degree = ally_counts + adv_counts
        max_degree = degree.max() if degree.max() > 0 else 1.0
        features[:, 8] = degree / max_degree

        # --- Feature 9: vote_entropy ---
        for idx in range(n_players):
            counts = vote_target_counts[idx]
            if not counts:
                continue
            total = sum(counts.values())
            probs = np.array(list(counts.values())) / total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features[idx, 9] = entropy

        # --- Normalize all features to [0, 1] within this tribal ---
        for col in range(features.shape[1]):
            col_min = features[:, col].min()
            col_max = features[:, col].max()
            if col_max > col_min:
                features[:, col] = (
                    (features[:, col] - col_min) / (col_max - col_min)
                )

        return features

    # ---- Build single graph ----------------------------------------------- #

    def build_tribal_graph(
        self,
        season: int,
        episode: int,
        order: int,
        tribal_idx: int,
        tribal_sequence: list[tuple[int, int, int]],
    ) -> Optional[Data]:
        """Build a PyG Data object for one tribal council.

        Returns:
            PyG Data with:
                x: (N, 10) node features
                edge_index: (2, E) edge connectivity
                edge_attr: (E, 2) [edge_weight, edge_type]
                y: (N,) binary labels — 1 for eliminated player, 0 otherwise
                season/episode/order: metadata
                player_names: list of player names (for interpretability)
            Returns None if the tribal has insufficient data.
        """
        _check_pyg()
        self._load_data()
        vh = self._vote_history

        players = self._get_players_at_tribal(season, episode, order)
        if len(players) < 3:
            # Degenerate tribal (final 2, fire challenge, etc.)
            return None

        # Who was eliminated
        tribal_votes = vh[
            (vh["season"] == season)
            & (vh["episode"] == episode)
            & (vh["order"] == order)
        ]
        if len(tribal_votes) == 0:
            return None
        voted_out = tribal_votes["voted_out"].iloc[0]
        if pd.isna(voted_out):
            return None

        player_to_idx = {p: i for i, p in enumerate(players)}

        # Node features
        x = self._build_node_features(
            season, episode, order, players, tribal_idx, tribal_sequence
        )

        # Edges
        edge_index, edge_weight, edge_type = self._build_edges(
            season, tribal_idx, tribal_sequence, players
        )

        # Labels: 1 for eliminated player, 0 for survivors
        y = np.zeros(len(players), dtype=np.float32)
        if voted_out in player_to_idx:
            y[player_to_idx[voted_out]] = 1.0

        # Build PyG Data object
        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.stack([
                torch.tensor(edge_weight, dtype=torch.float),
                torch.tensor(edge_type, dtype=torch.float),
            ], dim=1),
            y=torch.tensor(y, dtype=torch.float),
        )
        # Metadata (not used in forward pass, but useful for analysis)
        data.season = season
        data.episode = episode
        data.order = order
        data.player_names = players
        data.num_nodes = len(players)

        return data

    # ---- Build dataset for a set of seasons ------------------------------- #

    def build_season_graphs(
        self, season: int
    ) -> list[Data]:
        """Build all tribal council graphs for one season."""
        tribal_sequence = self._get_tribal_sequence(season)
        graphs = []
        for idx, (s, ep, ord_) in enumerate(tribal_sequence):
            graph = self.build_tribal_graph(s, ep, ord_, idx, tribal_sequence)
            if graph is not None:
                graphs.append(graph)
        return graphs

    def build_dataset(
        self,
        seasons: Optional[list[int]] = None,
        primary_only: bool = True,
    ) -> list[Data]:
        """Build the full training dataset across multiple seasons.

        Args:
            seasons: Specific seasons to include. If None, uses defaults.
            primary_only: If True and seasons is None, use only return
                seasons (8, 20, 34, 40). If False, use all seasons
                found in vote_history.csv.

        Returns:
            List of PyG Data objects, one per tribal council.
        """
        if seasons is None:
            if primary_only:
                seasons = self.RETURN_SEASONS
            else:
                # Discover all seasons from the data
                self._load_data()
                seasons = sorted(
                    int(s) for s in self._vote_history["season"].dropna().unique()
                )

        all_graphs = []
        for season in seasons:
            all_graphs.extend(self.build_season_graphs(season))
        return all_graphs


# =========================================================================== #
#                          GNN MODEL ARCHITECTURE                             #
# =========================================================================== #

class EliminationGAT(nn.Module):
    """Graph Attention Network for elimination prediction.

    Architecture:
    - 2-3 GAT layers with multi-head attention
    - Edge-type-aware: separate attention for alliance vs adversarial edges
    - Per-node output → softmax over remaining players at each tribal
    - Supports signed edge weights via edge_attr

    Why GAT over GraphSAGE:
    - Attention lets the model learn WHICH relationships matter most
      for predicting vulnerability (e.g., a single strong adversary
      may matter more than many weak allies)
    - Multi-head attention captures different relationship dynamics
      simultaneously (voting power, social threat, etc.)

    Input:
        x: (N, in_channels) node features
        edge_index: (2, E) edges
        edge_attr: (E, 2) [weight, type] — type 0=alliance, 1=adversarial

    Output:
        (N,) elimination logits per node (pre-softmax)
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 32,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int = 2,
    ):
        super().__init__()
        _check_pyg()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer: in_channels → hidden_channels * heads
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Middle GAT layers: hidden*heads → hidden*heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Final GAT layer: hidden*heads → hidden (single head, concat=False)
        if num_layers >= 2:
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Node-level classifier: hidden → 1 (elimination logit)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, 2) edge attributes [weight, type]
            batch: (N,) graph membership for batched graphs

        Returns:
            (N,) elimination logits per node
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Per-node elimination logit
        logits = self.classifier(x).squeeze(-1)
        return logits


# =========================================================================== #
#                              LOSS FUNCTIONS                                 #
# =========================================================================== #

class EliminationLoss(nn.Module):
    """Combined loss for elimination prediction.

    Combines:
    1. Focal cross-entropy: handles class imbalance (1 eliminated vs
       N-1 survivors per tribal). Focuses learning on hard examples.
    2. Pairwise ranking loss: the eliminated player should have higher
       elimination logit than every surviving player. This is a natural
       formulation — "is A more vulnerable than B?"

    The losses are weighted and summed. Default weights favor the
    pairwise ranking component since rank ordering matters more than
    calibrated probabilities for this task.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.3,
        pairwise_weight: float = 0.7,
        pairwise_margin: float = 1.0,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.pairwise_weight = pairwise_weight
        self.pairwise_margin = pairwise_margin

    def focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal cross-entropy loss over nodes in one graph.

        Args:
            logits: (N,) raw logits per node
            targets: (N,) binary — 1 for eliminated, 0 for survivor
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=0)
        # Probability assigned to the true class
        pt = torch.where(targets == 1, probs, 1 - probs)
        # Focal modulation
        focal_weight = (1 - pt) ** self.focal_gamma
        # Cross-entropy
        ce = -torch.where(
            targets == 1,
            torch.log(probs + 1e-8),
            torch.log(1 - probs + 1e-8),
        )
        return (focal_weight * ce).mean()

    def pairwise_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise margin ranking loss.

        The eliminated player's logit should exceed every survivor's
        logit by at least `margin`.

        Args:
            logits: (N,) raw logits per node
            targets: (N,) binary — 1 for eliminated
        """
        elim_mask = targets == 1
        surv_mask = targets == 0

        if elim_mask.sum() == 0 or surv_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        elim_logits = logits[elim_mask]  # (1,) typically
        surv_logits = logits[surv_mask]  # (N-1,)

        # Pairwise: eliminated should be higher than each survivor
        # loss = max(0, margin - (elim - surv))
        diffs = elim_logits.unsqueeze(1) - surv_logits.unsqueeze(0)
        loss = F.relu(self.pairwise_margin - diffs)
        return loss.mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss, handling batched graphs.

        When graphs are batched, computes loss per-graph (since softmax
        is over nodes within a single tribal) and averages.
        """
        if batch is None:
            # Single graph
            focal = self.focal_loss(logits, targets)
            pairwise = self.pairwise_loss(logits, targets)
            return self.focal_weight * focal + self.pairwise_weight * pairwise

        # Batched: compute per-graph loss
        total_loss = torch.tensor(0.0, device=logits.device)
        graph_ids = batch.unique()
        for gid in graph_ids:
            mask = batch == gid
            g_logits = logits[mask]
            g_targets = targets[mask]
            focal = self.focal_loss(g_logits, g_targets)
            pairwise = self.pairwise_loss(g_logits, g_targets)
            total_loss = total_loss + (
                self.focal_weight * focal + self.pairwise_weight * pairwise
            )
        return total_loss / len(graph_ids)


# =========================================================================== #
#                    SKLEARN-COMPATIBLE WRAPPER                               #
# =========================================================================== #

class GNNEliminationModel(SurvivorModel):
    """SurvivorModel wrapper for the GNN elimination predictor.

    This bridges two worlds:
    - The GNN operates on graphs (variable-size, relational)
    - The existing CV framework expects fit(X, y) / predict(X) with
      fixed-size feature matrices

    Bridging strategy:
    - fit() ignores X, y and instead trains on graph data built from
      vote_history for the specified seasons
    - predict(X) is not meaningful for graph-based prediction — this
      model provides predict_tribal() for graph-based inference and
      predict_season() for season-level elimination ordering
    - For compatibility with the existing evaluation framework,
      predict(X) returns placement estimates by training on graph data
      and mapping elimination order → placement

    The GNN is the most experimental model in the system. Its strength
    is modeling social dynamics that flat feature vectors can't capture.
    Its weakness is small sample size per graph (5-20 nodes) and the
    need to construct graphs at inference time.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        lr: float = 0.005,
        epochs: int = 100,
        batch_size: int = 8,
        window_size: int = 5,
        decay_rate: float = 0.7,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.3,
        pairwise_weight: float = 0.7,
        pairwise_margin: float = 1.0,
        seasons: Optional[tuple] = None,
        primary_only: bool = False,
        verbose: int = 0,
    ):
        """
        Args:
            hidden_channels: GNN hidden dimension.
            num_layers: Number of GAT layers.
            heads: Number of attention heads.
            dropout: Dropout rate.
            lr: Learning rate for Adam optimizer.
            epochs: Training epochs.
            batch_size: Number of graphs per training batch.
            window_size: Vote history window for graph construction.
            decay_rate: Time decay for edge weights.
            focal_gamma: Focal loss gamma parameter.
            focal_weight: Weight for focal loss component.
            pairwise_weight: Weight for pairwise ranking loss.
            pairwise_margin: Margin for pairwise ranking loss.
            seasons: Tuple of seasons to train on. None = use defaults.
            primary_only: If True and seasons is None, use only return
                seasons. If False, use all available seasons.
            verbose: 0=silent, 1=epoch summaries, 2=detailed.
        """
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.pairwise_weight = pairwise_weight
        self.pairwise_margin = pairwise_margin
        self.seasons = seasons
        self.primary_only = primary_only
        self.verbose = verbose

        self._model: Optional[EliminationGAT] = None
        self._graph_builder: Optional[TribalCouncilGraphBuilder] = None
        self._is_fitted = False

    def _build_graph_data(self) -> list[Data]:
        """Build graph dataset using the configured parameters."""
        self._graph_builder = TribalCouncilGraphBuilder(
            window_size=self.window_size,
            decay_rate=self.decay_rate,
        )
        season_list = list(self.seasons) if self.seasons else None
        return self._graph_builder.build_dataset(
            seasons=season_list,
            primary_only=self.primary_only,
        )

    def _train_on_graphs(self, graphs: list[Data]):
        """Train the GAT model on a list of tribal council graphs."""
        _check_pyg()

        if len(graphs) == 0:
            raise ValueError("No valid tribal council graphs to train on")

        # Determine input dimension from first graph
        in_channels = graphs[0].x.shape[1]

        # Fresh model each fit() call (fold isolation)
        self._model = EliminationGAT(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout,
        )

        criterion = EliminationLoss(
            focal_gamma=self.focal_gamma,
            focal_weight=self.focal_weight,
            pairwise_weight=self.pairwise_weight,
            pairwise_margin=self.pairwise_margin,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

        # Training loop
        self._model.train()
        n_graphs = len(graphs)

        for epoch in range(self.epochs):
            # Shuffle graphs each epoch
            perm = np.random.permutation(n_graphs)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_graphs, self.batch_size):
                batch_indices = perm[batch_start:batch_start + self.batch_size]
                batch_graphs = [graphs[i] for i in batch_indices]
                batch = Batch.from_data_list(batch_graphs)

                optimizer.zero_grad()
                logits = self._model(
                    batch.x, batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )
                loss = criterion(logits, batch.y, batch=batch.batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_norm=1.0
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step(avg_loss)

            if self.verbose >= 2:
                print(f"  Epoch {epoch+1}/{self.epochs} — loss: {avg_loss:.4f}")
            elif self.verbose >= 1 and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} — loss: {avg_loss:.4f}")

        self._is_fitted = True

    def fit(self, X, y) -> "GNNEliminationModel":
        """Train the GNN on graph data built from vote_history.

        NOTE: X and y are IGNORED — the GNN trains on graph-structured
        data built from vote_history.csv. This signature exists for
        compatibility with the sklearn interface and CV framework.

        The model builds graphs internally from the configured seasons.
        """
        _check_pyg()
        graphs = self._build_graph_data()
        self._train_on_graphs(graphs)
        return self

    def predict(self, X) -> np.ndarray:
        """Return placement estimates for compatibility.

        NOTE: For meaningful GNN-based predictions, use predict_tribal()
        or predict_season() instead. This method returns zeros as a
        placeholder since the GNN requires graph structure, not flat
        feature vectors.

        The GNN's real value is in predict_tribal() and predict_season()
        which operate on the social graph.
        """
        return np.zeros(len(X))

    def predict_tribal(self, graph: Data) -> dict[str, float]:
        """Predict elimination probabilities for a single tribal council.

        This is the GNN's native prediction interface.

        Args:
            graph: PyG Data object for one tribal council (from
                TribalCouncilGraphBuilder.build_tribal_graph).

        Returns:
            Dict mapping player name → elimination probability.
            Probabilities sum to 1.0 across remaining players.
        """
        _check_pyg()
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                graph.x, graph.edge_index, edge_attr=graph.edge_attr
            )
            probs = F.softmax(logits, dim=0).numpy()

        return {
            name: float(prob)
            for name, prob in zip(graph.player_names, probs)
        }

    def predict_season(self, season: int) -> list[dict]:
        """Predict elimination probabilities for every tribal in a season.

        Args:
            season: Season number to predict.

        Returns:
            List of dicts, one per tribal council, each containing:
                'episode': episode number
                'order': tribal order within episode
                'voted_out': actual elimination (ground truth)
                'predictions': dict of player → elimination probability
                'correct': whether the highest-probability player was
                    actually eliminated (top-1 accuracy)
        """
        _check_pyg()
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._graph_builder is None:
            self._graph_builder = TribalCouncilGraphBuilder(
                window_size=self.window_size,
                decay_rate=self.decay_rate,
            )

        results = []
        graphs = self._graph_builder.build_season_graphs(season)

        for graph in graphs:
            preds = self.predict_tribal(graph)
            voted_out = None
            for name, label in zip(graph.player_names, graph.y.numpy()):
                if label == 1.0:
                    voted_out = name
                    break

            most_vulnerable = max(preds, key=preds.get)
            results.append({
                "episode": graph.episode,
                "order": graph.order,
                "voted_out": voted_out,
                "predictions": preds,
                "correct": most_vulnerable == voted_out,
            })

        return results

    def evaluate_season(self, season: int) -> dict:
        """Evaluate the model on an entire season.

        Returns:
            Dict with:
                'top_1_accuracy': fraction of tribals where the model
                    correctly predicted the eliminated player
                'top_3_accuracy': fraction where eliminated player was
                    in the model's top 3 most vulnerable
                'mean_rank': average rank of the eliminated player in
                    the model's vulnerability ordering (1 = predicted
                    most vulnerable, N = predicted least vulnerable)
                'n_tribals': number of tribal councils evaluated
                'per_tribal': list of per-tribal prediction details
        """
        tribal_results = self.predict_season(season)
        if not tribal_results:
            return {
                "top_1_accuracy": 0.0,
                "top_3_accuracy": 0.0,
                "mean_rank": float("inf"),
                "n_tribals": 0,
                "per_tribal": [],
            }

        top_1_correct = 0
        top_3_correct = 0
        ranks = []

        for result in tribal_results:
            preds = result["predictions"]
            voted_out = result["voted_out"]
            if voted_out is None:
                continue

            # Sort by descending elimination probability
            sorted_players = sorted(
                preds.keys(), key=lambda p: preds[p], reverse=True
            )
            rank = sorted_players.index(voted_out) + 1 if voted_out in sorted_players else len(sorted_players)
            ranks.append(rank)

            if rank == 1:
                top_1_correct += 1
            if rank <= 3:
                top_3_correct += 1

        n = len(ranks)
        return {
            "top_1_accuracy": top_1_correct / n if n > 0 else 0.0,
            "top_3_accuracy": top_3_correct / n if n > 0 else 0.0,
            "mean_rank": float(np.mean(ranks)) if ranks else float("inf"),
            "n_tribals": n,
            "per_tribal": tribal_results,
        }

    def run_loso_cv(
        self,
        seasons: Optional[list[int]] = None,
        holdout_seasons: Optional[list[int]] = None,
        n_runs: int = 1,
    ) -> dict:
        """Leave-one-season-out cross-validation.

        For each fold, trains on graphs from all seasons except one,
        then evaluates elimination predictions on the held-out season.

        This is the honest evaluation — the model has never seen the
        held-out season's social dynamics during training.

        Args:
            seasons: Full pool of seasons to draw from. Default: all
                available (1-47).
            holdout_seasons: Which seasons to hold out for evaluation.
                If None, every season in `seasons` takes a turn.
                If set, only these seasons are held out — but training
                still uses all other seasons from the full pool.
                Use this to efficiently evaluate on specific seasons
                (e.g. return seasons) while training on all 47.
            n_runs: Number of runs to average over (accounts for
                training variance from random init / shuffle).

        Returns:
            Dict with:
                'per_season': dict of season → evaluation metrics
                'aggregate': mean metrics across all held-out seasons
                'summary_df': DataFrame with per-season results
        """
        _check_pyg()

        if self._graph_builder is None:
            self._graph_builder = TribalCouncilGraphBuilder(
                window_size=self.window_size,
                decay_rate=self.decay_rate,
            )

        if seasons is None:
            seasons = TribalCouncilGraphBuilder.ALL_SEASONS

        # Pre-build all graphs (expensive, so do it once)
        season_graphs: dict[int, list[Data]] = {}
        for s in seasons:
            graphs = self._graph_builder.build_season_graphs(s)
            if graphs:
                season_graphs[s] = graphs
        valid_seasons = sorted(season_graphs.keys())

        # Determine which seasons to hold out
        if holdout_seasons is not None:
            eval_seasons = [s for s in holdout_seasons if s in season_graphs]
        else:
            eval_seasons = valid_seasons

        if self.verbose >= 1:
            print(f"LOSO CV: {len(valid_seasons)} seasons in pool, "
                  f"{len(eval_seasons)} held-out folds, "
                  f"{sum(len(g) for g in season_graphs.values())} total graphs, "
                  f"{n_runs} run(s)")

        # Accumulate results across runs
        all_run_results: dict[int, list[dict]] = {
            s: [] for s in eval_seasons
        }

        for run in range(n_runs):
            if self.verbose >= 1 and n_runs > 1:
                print(f"\n--- Run {run+1}/{n_runs} ---")

            for held_out in eval_seasons:
                # Train on all seasons except held_out
                train_graphs = []
                for s in valid_seasons:
                    if s != held_out:
                        train_graphs.extend(season_graphs[s])

                if not train_graphs:
                    continue

                # Train a fresh model
                self._train_on_graphs(train_graphs)

                # Evaluate on held-out season
                eval_result = self.evaluate_season(held_out)
                all_run_results[held_out].append(eval_result)

                if self.verbose >= 1:
                    print(
                        f"  S{held_out:2d}: top1={eval_result['top_1_accuracy']:.1%} "
                        f"top3={eval_result['top_3_accuracy']:.1%} "
                        f"rank={eval_result['mean_rank']:.1f} "
                        f"({eval_result['n_tribals']} tribals)"
                    )

        # Average across runs per season
        per_season = {}
        for s in eval_seasons:
            results = all_run_results[s]
            if not results:
                continue
            per_season[s] = {
                "top_1_accuracy": np.mean([r["top_1_accuracy"] for r in results]),
                "top_3_accuracy": np.mean([r["top_3_accuracy"] for r in results]),
                "mean_rank": np.mean([r["mean_rank"] for r in results]),
                "n_tribals": results[0]["n_tribals"],
            }

        # Weighted aggregate (weight by number of tribals)
        total_tribals = sum(v["n_tribals"] for v in per_season.values())
        aggregate = {}
        if total_tribals > 0:
            for metric in ["top_1_accuracy", "top_3_accuracy", "mean_rank"]:
                aggregate[metric] = sum(
                    v[metric] * v["n_tribals"] for v in per_season.values()
                ) / total_tribals
        aggregate["total_tribals"] = total_tribals
        aggregate["n_seasons"] = len(per_season)

        # Build summary DataFrame
        rows = []
        for s, metrics in sorted(per_season.items()):
            rows.append({"season": s, **metrics})
        summary_df = pd.DataFrame(rows)

        return {
            "per_season": per_season,
            "aggregate": aggregate,
            "summary_df": summary_df,
        }

    @property
    def name(self) -> str:
        return (
            f"GNN_GAT_{self.num_layers}L_{self.hidden_channels}h"
            f"_{self.heads}head"
        )

    @property
    def description(self) -> str:
        return (
            f"Graph Attention Network for elimination prediction. "
            f"{self.num_layers} GAT layers, {self.hidden_channels} hidden, "
            f"{self.heads} attention heads. "
            f"Trains on vote_history social graph with "
            f"focal+pairwise loss."
        )


# =========================================================================== #
#                          MODEL FACTORY                                      #
# =========================================================================== #

def get_all_gnn_models() -> dict[str, GNNEliminationModel]:
    """Return a dict of preconfigured GNN model variants.

    Returns empty dict if PyG is not installed.
    """
    if not _PYG_AVAILABLE:
        return {}

    return {
        # Default: 2-layer GAT, moderate capacity
        "GNN_GAT_2L_32h": GNNEliminationModel(
            hidden_channels=32,
            num_layers=2,
            heads=4,
            dropout=0.3,
            epochs=100,
            primary_only=False,
        ),
        # Deeper: 3-layer GAT for capturing longer-range social dynamics
        "GNN_GAT_3L_64h": GNNEliminationModel(
            hidden_channels=64,
            num_layers=3,
            heads=4,
            dropout=0.3,
            epochs=150,
            primary_only=False,
        ),
        # Lightweight: fewer params, less overfitting risk on small graphs
        "GNN_GAT_2L_16h_light": GNNEliminationModel(
            hidden_channels=16,
            num_layers=2,
            heads=2,
            dropout=0.2,
            epochs=80,
            primary_only=False,
        ),
    }


# =========================================================================== #
#                        HYPERPARAMETER SWEEP                                 #
# =========================================================================== #

def run_gnn_sweep(
    seasons: Optional[list[int]] = None,
    holdout_seasons: Optional[list[int]] = None,
    n_runs: int = 1,
    epochs: int = 80,
    verbose: int = 1,
) -> pd.DataFrame:
    """Run a hyperparameter sweep with LOSO cross-validation.

    Tests combinations of graph construction params (window_size,
    decay_rate) and model architecture params (hidden_channels,
    num_layers, heads, loss weights).

    Args:
        seasons: Full pool of seasons for LOSO. Default: all (1-47).
        holdout_seasons: Which seasons to hold out for evaluation.
            If set, only these seasons are held out (one at a time),
            while training uses all other seasons from the pool.
            E.g. [8, 20, 34, 40] = 4 folds per config, each training
            on ~46 seasons. If None, every season gets held out.
        n_runs: Runs per config to average over.
        epochs: Training epochs per fold.
        verbose: 0=silent, 1=per-config summary, 2=per-season detail.

    Returns:
        DataFrame with one row per config, columns for params and
        aggregate LOSO metrics. Sorted by top_3_accuracy descending.
    """
    _check_pyg()

    configs = [
        # Vary graph construction
        {"window_size": 3, "decay_rate": 0.7, "hidden_channels": 32, "num_layers": 2, "heads": 4},
        {"window_size": 5, "decay_rate": 0.7, "hidden_channels": 32, "num_layers": 2, "heads": 4},
        {"window_size": 7, "decay_rate": 0.7, "hidden_channels": 32, "num_layers": 2, "heads": 4},
        {"window_size": 5, "decay_rate": 0.5, "hidden_channels": 32, "num_layers": 2, "heads": 4},
        {"window_size": 5, "decay_rate": 0.9, "hidden_channels": 32, "num_layers": 2, "heads": 4},
        # Vary architecture
        {"window_size": 5, "decay_rate": 0.7, "hidden_channels": 16, "num_layers": 2, "heads": 2},
        {"window_size": 5, "decay_rate": 0.7, "hidden_channels": 64, "num_layers": 2, "heads": 4},
        {"window_size": 5, "decay_rate": 0.7, "hidden_channels": 32, "num_layers": 3, "heads": 4},
        # Vary loss balance
        {"window_size": 5, "decay_rate": 0.7, "hidden_channels": 32, "num_layers": 2, "heads": 4,
         "focal_weight": 0.5, "pairwise_weight": 0.5},
        {"window_size": 5, "decay_rate": 0.7, "hidden_channels": 32, "num_layers": 2, "heads": 4,
         "focal_weight": 0.1, "pairwise_weight": 0.9},
    ]

    results = []
    for i, cfg in enumerate(configs):
        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Config {i+1}/{len(configs)}: {cfg}")
            print(f"{'='*60}")

        model = GNNEliminationModel(
            epochs=epochs,
            batch_size=8,
            lr=0.005,
            dropout=0.3,
            verbose=max(0, verbose - 1),
            **cfg,
        )

        loso = model.run_loso_cv(
            seasons=seasons,
            holdout_seasons=holdout_seasons,
            n_runs=n_runs,
        )

        agg = loso["aggregate"]
        row = {**cfg, **agg}
        results.append(row)

        if verbose >= 1:
            print(f"  => top1={agg.get('top_1_accuracy', 0):.1%} "
                  f"top3={agg.get('top_3_accuracy', 0):.1%} "
                  f"rank={agg.get('mean_rank', 0):.2f}")

    df = pd.DataFrame(results)
    df = df.sort_values("top_3_accuracy", ascending=False).reset_index(drop=True)
    return df
