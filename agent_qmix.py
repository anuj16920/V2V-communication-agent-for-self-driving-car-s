"""
QMIX Agent - Value Decomposition for Multi-Agent DQN

QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
Rashid et al., 2018

Key idea: Q_tot = mixing_network(Q_1, ..., Q_n, global_state)
- Individual agent Q-networks
- Mixing network combines them monotonically (preserves arg max)
- Centralized training, decentralized execution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Optional

from networks import TransformerDQN
from replay import PrioritizedReplayBuffer


class QMixingNetwork(nn.Module):
    """
    Monotonic mixing network that combines individual Q-values.
    Uses hypernetworks to generate weights based on global state.
    Ensures Q_tot is monotonic in each Q_i (preserves arg max).
    """
    
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        
        # Hypernetwork for first layer weights (always positive via abs)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim * n_agents, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents * mixing_embed_dim)
        )
        
        # Hypernetwork for first layer bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim * n_agents, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for second layer weights (always positive via abs)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim * n_agents, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetwork for second layer bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim * n_agents, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: (batch, n_agents) - individual Q-values
        states: (batch, n_agents, state_dim) - all agent states
        returns: (batch, 1) - mixed Q_total
        """
        batch_size = agent_qs.size(0)
        
        # Flatten global state
        global_state = states.reshape(batch_size, -1)  # (B, n_agents * state_dim)
        
        # Generate mixing network weights (ensure positive for monotonicity)
        w1 = torch.abs(self.hyper_w1(global_state))  # (B, n_agents * embed_dim)
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)  # (B, n_agents, embed_dim)
        
        b1 = self.hyper_b1(global_state)  # (B, embed_dim)
        
        w2 = torch.abs(self.hyper_w2(global_state))  # (B, embed_dim)
        w2 = w2.view(batch_size, self.embed_dim, 1)  # (B, embed_dim, 1)
        
        b2 = self.hyper_b2(global_state)  # (B, 1)
        
        # Mix Q-values
        # Layer 1: (B, n_agents, 1) @ (B, n_agents, embed_dim) -> (B, 1, embed_dim)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = torch.bmm(agent_qs, w1)  # (B, 1, embed_dim)
        hidden = hidden + b1.unsqueeze(1)  # Add bias
        hidden = F.elu(hidden)
        
        # Layer 2: (B, 1, embed_dim) @ (B, embed_dim, 1) -> (B, 1, 1)
        q_tot = torch.bmm(hidden, w2)  # (B, 1, 1)
        q_tot = q_tot + b2.unsqueeze(1)  # Add bias
        
        return q_tot.view(batch_size)  # (B,) — squeeze() collapses to scalar when B=1


class NStepBuffer:
    """N-step returns buffer (same as DQN)"""
    
    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        self.n = n_steps
        self.gamma = gamma
        self.buf: deque = deque(maxlen=n_steps)
    
    def push(self, transition: Tuple) -> Optional[Tuple]:
        self.buf.append(transition)
        if len(self.buf) < self.n:
            return None
        return self._make_nstep()
    
    def flush(self) -> List[Tuple]:
        transitions = []
        while len(self.buf) > 0:
            transitions.append(self._make_nstep())
            self.buf.popleft()
        return transitions
    
    def _make_nstep(self) -> Tuple:
        state0, actions0, _, _, _ = self.buf[0]
        _, _, _, state_n, done_n = self.buf[-1]
        
        G = 0.0
        for i, (_, _, rewards, _, d) in enumerate(self.buf):
            # rewards is array for all agents
            G += (self.gamma ** i) * np.mean(rewards)  # Average team reward
            if d:
                state_n = self.buf[i][3]
                done_n = True
                break
        
        return (state0, actions0, G, state_n, done_n)
    
    def reset(self):
        self.buf.clear()


class QMIXAgent:
    """
    QMIX Agent with Transformer-based individual Q-networks
    """
    
    def __init__(
        self,
        state_dim: int = 28,
        n_actions: int = 16,
        n_agents: int = 8,
        # Architecture
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        mixing_embed_dim: int = 64,   # 64 > 32: much more expressive hypernetwork
        # Training
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        n_steps: int = 3,
        tau: float = 0.005,
        grad_clip: float = 10.0,
        update_freq: int = 4,
        min_replay_size: int = 3000,  # warm-up before first gradient step
        # PER
        per_capacity: int = 200_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_steps: int = 150_000,
        # Exploration
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_steps: int = 100_000,  # faster decay than 150K
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.tau = tau
        self.grad_clip = grad_clip
        self.update_freq = update_freq
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.lr = lr
        self.min_replay_size = min_replay_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_steps = epsilon_steps
        
        # Individual Q-networks (shared across agents via Transformer)
        self.q_network = TransformerDQN(
            state_dim, n_actions, d_model, n_heads, n_layers, d_ff, dropout, noisy=False
        ).to(self.device)
        
        self.target_q_network = TransformerDQN(
            state_dim, n_actions, d_model, n_heads, n_layers, d_ff, dropout, noisy=False
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        # Mixing networks
        self.mixer = QMixingNetwork(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer = QMixingNetwork(n_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()
        
        # Optimizer for both Q-network and mixer
        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200_000, eta_min=lr * 0.1
        )
        
        # PER buffer
        self.per = PrioritizedReplayBuffer(
            capacity=per_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_steps=per_beta_steps,
        )
        
        # N-step buffer
        self.nstep_buf = NStepBuffer(n_steps, gamma)
        
        self._train_steps = 0
        self._env_steps = 0
    
    def _get_epsilon(self) -> float:
        frac = min(1.0, self._env_steps / self.epsilon_steps)
        # Polynomial decay: stays high longer early, then drops sharply — better exploration
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - frac) ** 2
    
    @torch.no_grad()
    def act(self, states: np.ndarray) -> np.ndarray:
        """Decentralized execution: each agent picks action based on local obs"""
        epsilon = self._get_epsilon()
        actions = np.zeros(self.n_agents, dtype=np.int64)
        
        # Get Q-values for all agents
        s = torch.FloatTensor(states).unsqueeze(0).to(self.device)  # (1, N, S)
        self.q_network.eval()
        q_all = self.q_network(s).squeeze(0)  # (N, A)
        
        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[i] = np.random.randint(0, self.n_actions)
            else:
                actions[i] = q_all[i].argmax().item()
        
        return actions
    
    def store(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ):
        """Store transition with n-step returns"""
        self._env_steps += 1
        
        t = (states, actions, rewards, next_states, done)
        ready = self.nstep_buf.push(t)
        if ready is not None:
            self.per.push(*ready)
        
        if done:
            for t in self.nstep_buf.flush():
                self.per.push(*t)
            self.nstep_buf.reset()
    
    def clear_buffer(self):
        """Clear PER and n-step buffers — call on curriculum stage transitions."""
        self.per.clear()
        self.nstep_buf.reset()

    def reset_optimizer(self, lr: float = None):
        """Warm-restart optimizer and LR scheduler after a curriculum transition."""
        if lr is None:
            lr = self.lr
        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100_000, eta_min=lr * 0.1
        )

    def train_step(self) -> float:
        if (
            len(self.per) < self.min_replay_size
            or self._env_steps % self.update_freq != 0
        ):
            return 0.0
        
        batch = self.per.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors
        s = torch.FloatTensor(states).to(self.device)  # (B, N, S)
        ns = torch.FloatTensor(next_states).to(self.device)  # (B, N, S)
        a = torch.LongTensor(actions).to(self.device)  # (B, N)
        r = torch.FloatTensor(rewards).to(self.device)  # (B,) - team reward
        d = torch.FloatTensor(dones).to(self.device)  # (B,)
        w = torch.FloatTensor(weights).to(self.device)  # (B,)
        
        # Get individual Q-values
        self.q_network.train()
        q_vals = self.q_network(s)  # (B, N, A)
        
        # Gather chosen actions
        chosen_q = q_vals.gather(2, a.unsqueeze(2)).squeeze(2)  # (B, N)
        
        # Mix to get Q_tot
        q_tot = self.mixer(chosen_q, s)  # (B,)
        
        # Target Q_tot
        with torch.no_grad():
            self.q_network.eval()
            next_q_online = self.q_network(ns)  # (B, N, A)
            best_actions = next_q_online.argmax(dim=2)  # (B, N)
            
            self.target_q_network.eval()
            next_q_target = self.target_q_network(ns)  # (B, N, A)
            next_q = next_q_target.gather(2, best_actions.unsqueeze(2)).squeeze(2)  # (B, N)
            
            next_q_tot = self.target_mixer(next_q, ns)  # (B,)
            
            gamma_n = self.gamma ** self.n_steps
            target_q_tot = r + gamma_n * next_q_tot * (1.0 - d)
        
        # TD error with importance sampling weights
        td_error = q_tot - target_q_tot
        loss = (w * F.huber_loss(q_tot, target_q_tot, reduction="none")).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            self.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        
        # Update PER priorities
        new_prios = td_error.detach().abs().cpu().numpy()
        self.per.update_priorities(indices, new_prios)
        
        # Soft update target networks
        with torch.no_grad():
            for tp, op in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(op.data * self.tau)
            for tp, op in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(op.data * self.tau)
        
        self._train_steps += 1
        return float(loss.item())
    
    def param_count(self) -> int:
        q_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        mixer_params = sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
        return q_params + mixer_params
