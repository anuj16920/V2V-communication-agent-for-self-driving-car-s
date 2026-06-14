"""
Multi-Agent PPO (MAPPO) Agent — SOTA version

Improvements over original:
  - Orthogonal weight initialization (critical for PPO stability)
  - Value function clipping (prevents large value jumps)
  - Per-episode reward normalization before GAE
  - Entropy coefficient decay (0.05 → 0.005 over training)
  - Fewer update epochs (4 vs 15 — on-policy constraint)
  - Cosine annealing LR schedulers
  - MAPPO curriculum support (set_curriculum_stage)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict

from networks import MAPPOActor, MAPPOCritic


def _ortho_init(module: nn.Module, gain: float = np.sqrt(2)):
    """Apply orthogonal initialization to all Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.constant_(m.bias, 0.0)


class RolloutBuffer:
    """Stores one episode of transitions for PPO update."""

    def __init__(self):
        self.states      : List = []
        self.actions     : List = []
        self.log_probs   : List = []
        self.rewards     : List = []
        self.values      : List = []
        self.dones       : List = []
        self.global_states: List = []

    def push(self, state, action, log_prob, reward, value, done, global_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.global_states.append(global_state)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class MAPPOAgent:
    """
    MAPPO: Multi-Agent PPO with centralized critic.

    One shared actor + one centralized critic.
    All agents share the actor parameters (parameter sharing).
    Critic sees global state (all agents concatenated).
    """

    def __init__(
        self,
        state_dim:    int   = 28,
        n_actions:    int   = 16,
        n_agents:     int   = 8,
        d_model:      int   = 128,
        # PPO hyperparams
        lr_actor:     float = 3e-4,
        lr_critic:    float = 1e-3,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_eps:     float = 0.2,
        entropy_coef: float = 0.01,
        entropy_min:  float = 0.001,   # minimum after decay
        value_coef:   float = 0.5,
        epochs:       int   = 4,       # 4 not 15 — on-policy constraint
        minibatch:    int   = 32,
        grad_clip:    float = 0.5,
        total_episodes: int = 3000,    # for entropy decay schedule
        device:       str   = "cuda",
    ):
        self.state_dim    = state_dim
        self.n_actions    = n_actions
        self.n_agents     = n_agents
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.entropy_min  = entropy_min
        self.value_coef   = value_coef
        self.epochs       = epochs
        self.minibatch    = minibatch
        self.grad_clip    = grad_clip
        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self._episode     = 0
        self._total_episodes = total_episodes

        self.actor  = MAPPOActor(state_dim, n_actions, d_model).to(self.device)
        self.critic = MAPPOCritic(state_dim, n_agents, d_model * 2).to(self.device)

        # Orthogonal initialization — critical for PPO stability
        _ortho_init(self.actor,  gain=np.sqrt(2))
        _ortho_init(self.critic, gain=np.sqrt(2))
        # Output layers use smaller gain
        nn.init.orthogonal_(list(self.actor.net.children())[-1].weight, gain=0.01)
        nn.init.orthogonal_(list(self.critic.net.children())[-1].weight, gain=1.0)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor,  eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        self.actor_scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt,  T_max=total_episodes, eta_min=lr_actor  * 0.1
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_opt, T_max=total_episodes, eta_min=lr_critic * 0.1
        )

        self.buffer = RolloutBuffer()

    def _current_entropy_coef(self) -> float:
        """Decay entropy coefficient linearly over training."""
        frac = min(1.0, self._episode / max(1, self._total_episodes))
        return max(self.entropy_min, self.entropy_coef * (1.0 - frac) + self.entropy_min * frac)

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, states: np.ndarray) -> np.ndarray:
        """states: (N, state_dim) → actions (N,)"""
        s = torch.FloatTensor(states).to(self.device)
        dist    = self.actor.get_dist(s)
        actions = dist.sample()
        return actions.cpu().numpy()

    @torch.no_grad()
    def act_with_info(self, states: np.ndarray):
        """Returns actions, log_probs, values for rollout storage."""
        s        = torch.FloatTensor(states).to(self.device)
        global_s = s.reshape(1, -1)

        dist      = self.actor.get_dist(s)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions)
        values    = self.critic(global_s).squeeze()

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.item(),
        )

    # ──────────────────────────────────────────────────────────────────
    # Store
    # ──────────────────────────────────────────────────────────────────
    def store(self, states, actions, log_probs, rewards, value, done):
        global_state = states.reshape(-1)
        self.buffer.push(
            states.copy(), actions.copy(), log_probs.copy(),
            float(np.mean(rewards)), float(value), float(done), global_state.copy()
        )

    # ──────────────────────────────────────────────────────────────────
    # Train (called at episode end)
    # ──────────────────────────────────────────────────────────────────
    def train(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {}

        self._episode += 1

        # ── Build arrays ─────────────────────────────────────────────
        states        = np.array(self.buffer.states)         # (T, N, S)
        actions       = np.array(self.buffer.actions)        # (T, N)
        old_log_probs = np.array(self.buffer.log_probs)      # (T, N)
        rewards_raw   = np.array(self.buffer.rewards)        # (T,)
        values_raw    = np.array(self.buffer.values)         # (T,)
        dones         = np.array(self.buffer.dones)          # (T,)
        global_states = np.array(self.buffer.global_states)  # (T, N*S)

        T = len(rewards_raw)

        # Per-episode reward normalization — critical for stable value estimation
        if rewards_raw.std() > 1e-8:
            rewards_raw = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)

        # ── GAE ───────────────────────────────────────────────────────
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0
        for t in reversed(range(T)):
            next_val = values_raw[t + 1] if t + 1 < T else 0.0
            delta    = rewards_raw[t] + self.gamma * next_val * (1.0 - dones[t]) - values_raw[t]
            gae      = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns    = advantages + values_raw
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Convert to tensors ────────────────────────────────────────
        TN = T * self.n_agents
        s_flat     = torch.FloatTensor(states.reshape(TN, self.state_dim)).to(self.device)
        a_flat     = torch.LongTensor(actions.reshape(TN)).to(self.device)
        old_lp     = torch.FloatTensor(old_log_probs.reshape(TN)).to(self.device)
        adv_t      = torch.FloatTensor(np.repeat(advantages, self.n_agents)).to(self.device)
        ret_t      = torch.FloatTensor(np.repeat(returns,    self.n_agents)).to(self.device)

        gs_t          = torch.FloatTensor(global_states).to(self.device)   # (T, N*S)
        ret_scalar    = torch.FloatTensor(returns).to(self.device)          # (T,)
        old_val_scalar= torch.FloatTensor(values_raw).to(self.device)       # (T,) for value clip

        ent_coef = self._current_entropy_coef()
        logs = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        for _ in range(self.epochs):
            # ── Actor update ─────────────────────────────────────────
            perm = torch.randperm(TN)
            for start in range(0, TN, self.minibatch):
                idx    = perm[start:start + self.minibatch]
                dist   = self.actor.get_dist(s_flat[idx])
                new_lp = dist.log_prob(a_flat[idx])
                entropy= dist.entropy().mean()

                ratio  = (new_lp - old_lp[idx]).exp()
                adv_mb = adv_t[idx]
                surr1  = ratio * adv_mb
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_opt.step()

                logs["actor_loss"] += actor_loss.item()
                logs["entropy"]    += entropy.item()

            # ── Critic update with value clipping ────────────────────
            perm_c = torch.randperm(T)
            for start in range(0, T, self.minibatch):
                idx_c    = perm_c[start:start + self.minibatch]
                val_pred = self.critic(gs_t[idx_c]).squeeze(1)

                # PPO-style value clipping prevents large value jumps
                val_clipped = old_val_scalar[idx_c] + (
                    val_pred - old_val_scalar[idx_c]
                ).clamp(-self.clip_eps, self.clip_eps)
                crit_loss = torch.max(
                    F.mse_loss(val_pred,   ret_scalar[idx_c]),
                    F.mse_loss(val_clipped, ret_scalar[idx_c]),
                )

                self.critic_opt.zero_grad()
                crit_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_opt.step()

                logs["critic_loss"] += crit_loss.item()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.buffer.clear()
        return logs

    def param_count(self) -> int:
        return (
            sum(p.numel() for p in self.actor.parameters()  if p.requires_grad)
            + sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        )
