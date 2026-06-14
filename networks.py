"""
Neural network architectures for V2V MARL

Components:
  - NoisyLinear      : learnable noise for exploration (replaces ε-greedy)
  - TransformerEncoder: multi-head self-attention over agent states
  - DuelingHead      : dueling advantage + value decomposition
  - TransformerDQN   : full Dueling Double DQN with Transformer backbone
  - MAPPOActor       : shared actor for MAPPO
  - MAPPOCritic      : centralized critic for MAPPO (sees global state)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
# NoisyLinear
# ─────────────────────────────────────────────
class NoisyLinear(nn.Module):
    """
    Factorised Noisy Linear layer.
    Fortunato et al., 2017 "Noisy Networks for Exploration".
    Replaces ε-greedy exploration with learned stochastic weights.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_i = self._scale_noise(self.in_features)
        eps_j = self._scale_noise(self.out_features)
        # Use non-inplace assignment to avoid autograd issues
        self.weight_epsilon = eps_j.outer(eps_i).to(self.weight_mu.device)
        self.bias_epsilon   = eps_j.to(self.weight_mu.device)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ─────────────────────────────────────────────
# Transformer Encoder
# ─────────────────────────────────────────────
class AgentTransformerEncoder(nn.Module):
    """
    Processes all agents' observations jointly via multi-head self-attention.

    Input : (batch, n_agents, state_dim)
    Output: (batch, n_agents, d_model)

    Each agent's embedding attends to all other agents → captures
    inter-vehicle interference patterns and coordination signals.
    """

    def __init__(
        self,
        state_dim:  int = 28,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 2,
        d_ff:       int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos_emb    = nn.Parameter(torch.zeros(1, 64, d_model))  # up to 64 agents

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ff,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, state_dim)
        B, N, _ = x.shape
        h = self.input_proj(x)                          # (B, N, d_model)
        h = h + self.pos_emb[:, :N, :]                 # learnable positional embedding
        h = self.transformer(h)                         # (B, N, d_model)
        return self.out_norm(h)                         # (B, N, d_model)


# ─────────────────────────────────────────────
# Dueling Head
# ─────────────────────────────────────────────
class DuelingHead(nn.Module):
    """
    Dueling network: V(s) + A(s,a) - mean(A)
    Wang et al., 2016 "Dueling Network Architectures"
    """

    def __init__(self, d_model: int, n_actions: int, noisy: bool = True):
        super().__init__()
        Linear = NoisyLinear if noisy else nn.Linear

        self.value_stream = nn.Sequential(
            Linear(d_model, 128),
            nn.ReLU(),
            Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            Linear(d_model, 128),
            nn.ReLU(),
            Linear(128, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        val = self.value_stream(x)              # (B, 1)
        adv = self.adv_stream(x)               # (B, n_actions)
        return val + adv - adv.mean(dim=-1, keepdim=True)

    def sample_noise(self):
        """Resample noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


# ─────────────────────────────────────────────
# Full Transformer-Dueling DQN
# ─────────────────────────────────────────────
class TransformerDQN(nn.Module):
    """
    Shared-parameter Dueling Double DQN with Transformer backbone.

    Forward pass:
      1. All agents' states fed through shared Transformer → context-aware embeddings
      2. Each agent's embedding → Dueling head → Q-values

    Parameter sharing across agents + attention = handles any density.
    """

    def __init__(
        self,
        state_dim: int   = 28,
        n_actions: int   = 16,
        d_model:   int   = 128,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        d_ff:      int   = 256,
        dropout:   float = 0.1,
        noisy:     bool  = True,
    ):
        super().__init__()
        self.encoder = AgentTransformerEncoder(state_dim, d_model, n_heads, n_layers, d_ff, dropout)
        self.head    = DuelingHead(d_model, n_actions, noisy=noisy)

    def forward(self, states: Tensor) -> Tensor:
        """
        states: (B, N, state_dim)  — B=batch, N=n_agents
        returns: (B, N, n_actions) Q-values
        """
        enc  = self.encoder(states)          # (B, N, d_model)
        B, N, D = enc.shape
        flat = enc.reshape(B * N, D)         # (B*N, d_model)
        q    = self.head(flat)               # (B*N, n_actions)
        return q.reshape(B, N, -1)           # (B, N, n_actions)

    def sample_noise(self):
        self.head.sample_noise()


# ─────────────────────────────────────────────
# MAPPO Networks
# ─────────────────────────────────────────────
class MAPPOActor(nn.Module):
    """
    Shared actor: maps single agent's local obs → action logits.
    LayerNorm added after each hidden layer for gradient stability.
    """

    def __init__(self, state_dim: int = 28, n_actions: int = 16, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)   # logits

    def get_dist(self, x: Tensor):
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)


class MAPPOCritic(nn.Module):
    """
    Centralized critic: sees ALL agents' observations concatenated.
    Input: (B, N * state_dim)   — global state
    Output: (B, 1) state value
    """

    def __init__(self, state_dim: int = 28, n_agents: int = 8, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * n_agents, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, global_state: Tensor) -> Tensor:
        return self.net(global_state)
