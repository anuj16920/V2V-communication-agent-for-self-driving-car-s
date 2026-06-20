# -*- coding: utf-8 -*-
"""
Predictive Graph Attention Networks for V2V MARL
==================================================
Novel architecture: PGAT-MAPPO
  - Dynamic vehicle graph built from Manhattan road topology
  - Edge features physically motivated by LoS/NLoS geometry
  - Predictive encoding: current + future (k steps ahead) graphs
  - N-agnostic critic via graph pooling (works n=8 to n=120, same weights)

Graph structure:
  Nodes : vehicles  (N)
  Edges : pairs within MAX_COMM_RANGE (500 m)
  Edge features (7-dim):
    [0] distance_norm       -- proximity
    [1] los_flag            -- building occlusion
    [2] heading_alignment   -- movement correlation
    [3] channel_overlap     -- currently interfering
    [4] sinr_ratio          -- relative link quality
    [5] future_los          -- will become LoS in k steps?
    [6] intersection_risk   -- both vehicles approaching same intersection
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Constants (must match env_manhattan.py) ───────────────────────────────────
MAX_COMM_RANGE  = 500.0
K_PREDICT       = 3        # look-ahead steps
DT              = 0.1      # seconds per step
EDGE_DIM        = 7
AREA_X          = 1332.5 + 7.5   # ~1340 m
AREA_Y          = 1332.5 + 7.5

# Approximate street positions (from env_manhattan.py)
_GRID_STEP = 265.0
_H_STREETS = np.array([7.5 + i * _GRID_STEP for i in range(6)])
_V_STREETS = np.array([7.5 + i * _GRID_STEP for i in range(6)])
_INTR_R    = 20.0
_STREET_W  = 15.0          # on-street tolerance = _STREET_W / 2 (matches env)

# Flattened intersection grid points (36, 2) for vectorised distance queries
_INTERSECTIONS = np.array([[vx, hy] for vx in _V_STREETS for hy in _H_STREETS],
                          dtype=np.float32)

# ── Graph construction (fully vectorised) ─────────────────────────────────────

def _los_matrix_vec(pos: np.ndarray) -> np.ndarray:
    """
    Vectorised LoS matrix from positions. (N, N) bool.
    Matches env_manhattan._is_los: LoS iff within 500 m AND (within intersection
    radius OR on the same horizontal/vertical street segment).
    """
    N = len(pos)
    x, y = pos[:, 0], pos[:, 1]
    dh = np.abs(y[:, None] - _H_STREETS[None, :])   # (N, 6)
    dv = np.abs(x[:, None] - _V_STREETS[None, :])   # (N, 6)
    h_idx = dh.argmin(1); on_h = dh.min(1) < _STREET_W / 2.0
    v_idx = dv.argmin(1); on_v = dv.min(1) < _STREET_W / 2.0

    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)   # (N, N)
    same_h = on_h[:, None] & on_h[None, :] & (h_idx[:, None] == h_idx[None, :])
    same_v = on_v[:, None] & on_v[None, :] & (v_idx[:, None] == v_idx[None, :])
    los = (D <= MAX_COMM_RANGE) & ((D < _INTR_R) | same_h | same_v)
    return los


def _nearest_intersection_id(pos: np.ndarray) -> np.ndarray:
    """Per-node nearest-intersection index (-1 if none within 3*INTR_R). (N,)."""
    d = np.linalg.norm(pos[:, None, :] - _INTERSECTIONS[None, :, :], axis=2)  # (N,36)
    nearest = d.argmin(1)
    within  = d[np.arange(len(pos)), nearest] < _INTR_R * 3.0
    return np.where(within, nearest, -1)


def build_graph(positions: np.ndarray,
                velocities: np.ndarray,
                los_matrix: np.ndarray,
                sinrs: np.ndarray,
                channels: np.ndarray,
                k_steps: int = 0) -> tuple:
    """
    Build adjacency and edge feature tensors from environment state. Vectorised.

    Args:
        positions  : (N, 2) current positions in metres
        velocities : (N, 2) velocity vectors m/s
        los_matrix : (N, N) bool current LoS (used as ground truth when k_steps=0)
        sinrs      : (N,) current SINR values
        channels   : (N,) current channel selections (0-3)
        k_steps    : prediction horizon (0 = current graph, K_PREDICT = future graph)

    Returns:
        adj     : (N, N) float adjacency (1 if edge exists, excluding self-loop)
        edge_ft : (N, N, EDGE_DIM) edge feature matrix
    """
    N = len(positions)
    positions = positions.astype(np.float32)

    if k_steps > 0:
        pred_pos = positions + velocities * DT * k_steps
        pred_pos[:, 0] = np.clip(pred_pos[:, 0], _V_STREETS[0], _V_STREETS[-1])
        pred_pos[:, 1] = np.clip(pred_pos[:, 1], _H_STREETS[0], _H_STREETS[-1])
    else:
        pred_pos = positions

    # ── Adjacency (within comm range, no self-loop) ──────────────────────────
    D   = np.linalg.norm(pred_pos[:, None, :] - pred_pos[None, :, :], axis=2)  # (N,N)
    eye = np.eye(N, dtype=bool)
    adj = ((D <= MAX_COMM_RANGE) & ~eye).astype(np.float32)

    # ── [0] normalised distance ──────────────────────────────────────────────
    d_norm = (D / MAX_COMM_RANGE).astype(np.float32)

    # ── [1] LoS flag (env ground truth for current, recompute for future) ────
    if k_steps == 0:
        los = los_matrix.astype(np.float32)
    else:
        los = _los_matrix_vec(pred_pos).astype(np.float32)

    # ── [2] heading alignment (cosine of unit direction vectors) ─────────────
    spd      = np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8
    dir_unit = velocities / spd                              # (N, 2)
    heading  = (dir_unit @ dir_unit.T).astype(np.float32)    # (N, N)

    # ── [3] channel overlap ──────────────────────────────────────────────────
    ch_overlap = (channels[:, None] == channels[None, :]).astype(np.float32)

    # ── [4] SINR ratio (tanh of log ratio) ───────────────────────────────────
    log_sinr   = np.log(np.maximum(sinrs, 1e-9))
    sinr_ratio = np.tanh(log_sinr[:, None] - log_sinr[None, :]).astype(np.float32)

    # ── [5] future LoS (K_PREDICT steps ahead of this graph) ─────────────────
    if k_steps == 0:
        fut_pos = pred_pos + velocities * DT * K_PREDICT
        fut_pos[:, 0] = np.clip(fut_pos[:, 0], _V_STREETS[0], _V_STREETS[-1])
        fut_pos[:, 1] = np.clip(fut_pos[:, 1], _H_STREETS[0], _H_STREETS[-1])
        future_los = _los_matrix_vec(fut_pos).astype(np.float32)
    else:
        future_los = los

    # ── [6] intersection risk (both near the SAME intersection) ──────────────
    intr_id   = _nearest_intersection_id(pred_pos)           # (N,)
    same_intr = (intr_id[:, None] == intr_id[None, :]) & (intr_id[:, None] >= 0)
    intr_risk = same_intr.astype(np.float32)

    # ── Stack edge features and mask out non-edges ───────────────────────────
    edge_ft = np.stack([d_norm, los, heading, ch_overlap,
                        sinr_ratio, future_los, intr_risk], axis=-1)  # (N,N,7)
    edge_ft *= adj[..., None]                                 # zero non-edges
    return adj, edge_ft


def graph_from_info(info: dict, device: str = "cpu") -> tuple:
    """
    Convenience wrapper: build (current, future) graph tensors from env info dict.
    Returns:
        adj_cur, eft_cur   : current graph
        adj_fut, eft_fut   : predicted K_PREDICT-step future graph
    """
    pos  = info["positions"]
    vel  = info["velocities"]
    los  = info["los_matrix"]
    sinr = info["sinrs"]
    ch   = info["channels"]

    adj_c, eft_c = build_graph(pos, vel, los, sinr, ch, k_steps=0)
    adj_f, eft_f = build_graph(pos, vel, los, sinr, ch, k_steps=K_PREDICT)

    to = lambda x: torch.FloatTensor(x).to(device)
    return to(adj_c), to(eft_c), to(adj_f), to(eft_f)


# ── Graph Attention Layer ─────────────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """
    Multi-head graph attention with edge feature bias.

    Attention score: a_ij = softmax( (q_i . k_j) / sqrt(d) + W_e * e_ij )

    Physical interpretation:
      - LoS edges (e[1]=1) learn higher base attention weight
      - Channel-overlapping pairs (e[3]=1) attend more for interference coord.
      - Future-LoS pairs (e[5]=1) prepare proactively
    """

    def __init__(self, d_model: int, n_heads: int, edge_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        # Edge bias: maps edge features to per-head scalar bias
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, n_heads * 4),
            nn.ReLU(),
            nn.Linear(n_heads * 4, n_heads),
        )

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor, edge_ft: Tensor) -> Tensor:
        """
        Batched graph attention.
        x       : (B, N, d_model)
        adj     : (B, N, N)        -- 1 if edge exists
        edge_ft : (B, N, N, edge_dim)
        Returns : (B, N, d_model)
        """
        B, N, _ = x.shape
        residual = x
        x = self.norm1(x)

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k)   # (B, N, H, d_k)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k)

        # Scaled dot-product: (B, N_q, N_k, H)  i=query, j=key
        scores = torch.einsum("bihd,bjhd->bijh", Q, K) / math.sqrt(self.d_k)

        # Edge bias from physical edge features: (B, N, N, H)
        scores = scores + self.edge_bias(edge_ft)

        # Mask non-edges (keep self-loop for numerical stability)
        eye   = torch.eye(N, device=adj.device).unsqueeze(0)   # (1, N, N)
        mask  = (adj + eye).unsqueeze(-1)                      # (B, N, N, 1)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=2)                        # normalise over keys
        attn = self.drop(attn)

        # Aggregate: (B, N, H, d_k) -> (B, N, d_model)
        out = torch.einsum("bijh,bjhd->bihd", attn, V).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)
        x   = residual + self.drop(out)

        # FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ── Predictive Graph Encoder ──────────────────────────────────────────────────

class PredictiveGraphEncoder(nn.Module):
    """
    Encodes both current and predicted-future vehicle graphs.

    Current graph  -> encodes present interference patterns
    Future graph   -> encodes upcoming topology changes (intersection crossings)

    Both streams are fused: agents can proactively allocate channels
    before a new LoS link (potential interferer) appears.
    """

    def __init__(self, state_dim: int = 32, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 edge_dim: int = EDGE_DIM, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Node embedding
        self.node_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Two separate stacks: current and future
        self.cur_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, n_heads, edge_dim, dropout)
            for _ in range(n_layers)
        ])
        self.fut_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, n_heads, edge_dim, dropout)
            for _ in range(n_layers)
        ])

        # Fusion: current + future -> d_model
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor,
                adj_cur: Tensor, eft_cur: Tensor,
                adj_fut: Tensor, eft_fut: Tensor) -> Tensor:
        """
        obs     : (B, N, state_dim)
        adj_*   : (B, N, N)
        eft_*   : (B, N, N, EDGE_DIM)
        Returns : (B, N, d_model) fused node embeddings
        """
        h = self.node_proj(obs)           # (B, N, d_model)

        # Current graph stream
        h_c = h
        for layer in self.cur_layers:
            h_c = layer(h_c, adj_cur, eft_cur)

        # Future graph stream
        h_f = h
        for layer in self.fut_layers:
            h_f = layer(h_f, adj_fut, eft_fut)

        # Fuse: agents know current state AND what's coming
        return self.fusion(torch.cat([h_c, h_f], dim=-1))   # (N, d_model)


# ── Actor and Critic ──────────────────────────────────────────────────────────

class GraphMAPPOActor(nn.Module):
    """
    Graph-aware actor.
    Each agent's action is conditioned on its neighbors' states via attention.
    """

    def __init__(self, state_dim: int = 32, n_actions: int = 16,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.encoder = PredictiveGraphEncoder(state_dim, d_model,
                                              n_heads, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_actions),
        )
        # Orthogonal init (skip bias for bias-free attention projections)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Smaller gain on output
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)

    def forward(self, obs, adj_cur, eft_cur, adj_fut, eft_fut):
        """obs: (B, N, state_dim) -> logits (B, N, n_actions)."""
        emb = self.encoder(obs, adj_cur, eft_cur, adj_fut, eft_fut)
        return self.head(emb)

    def get_dist(self, obs, adj_cur, eft_cur, adj_fut, eft_fut):
        logits = self.forward(obs, adj_cur, eft_cur, adj_fut, eft_fut)
        return torch.distributions.Categorical(logits=logits)


class GraphMAPPOCritic(nn.Module):
    """
    N-agnostic centralized critic using graph mean pooling.
    Works for ANY number of vehicles with the SAME weights.

    Standard MAPPO critic: input = state_dim * n_agents (fixed N!)
    This critic: global mean pool of node embeddings (any N)
    """

    def __init__(self, state_dim: int = 32, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.encoder = PredictiveGraphEncoder(state_dim, d_model,
                                              n_heads, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.head[-1].weight, gain=1.0)

    def forward(self, obs, adj_cur, eft_cur, adj_fut, eft_fut):
        """obs: (B, N, state_dim) -> value (B,). Mean-pool over nodes (any N)."""
        emb  = self.encoder(obs, adj_cur, eft_cur, adj_fut, eft_fut)
        pool = emb.mean(dim=1)                    # (B, d_model) -- any N
        return self.head(pool).squeeze(-1)        # (B,)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
