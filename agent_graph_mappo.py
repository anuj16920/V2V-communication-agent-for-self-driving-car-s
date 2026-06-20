# -*- coding: utf-8 -*-
"""
PGAT-MAPPO Agent
=================
Predictive Graph Attention MAPPO for V2V resource allocation.

Key differences from standard MAPPO:
  1. Actor and critic use graph attention (not MLP)
  2. Both current AND future graphs fed at every step
  3. Critic is N-agnostic (mean pool) -> single model for n=8 to n=120
  4. Graph data comes from env info dict (positions, velocities, los_matrix)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict

from networks_graph import GraphMAPPOActor, GraphMAPPOCritic, graph_from_info


class GraphRolloutBuffer:
    def __init__(self):
        self.obs        : List = []
        self.actions    : List = []
        self.log_probs  : List = []
        self.rewards    : List = []
        self.values     : List = []
        self.dones      : List = []
        self.adj_cur    : List = []
        self.eft_cur    : List = []
        self.adj_fut    : List = []
        self.eft_fut    : List = []

    def push(self, obs, actions, log_probs, reward, value, done,
             adj_c, eft_c, adj_f, eft_f):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.adj_cur.append(adj_c)
        self.eft_cur.append(eft_c)
        self.adj_fut.append(adj_f)
        self.eft_fut.append(eft_f)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


class PGATMAPPOAgent:
    """
    Predictive Graph Attention MAPPO Agent.

    At each step it receives:
      - obs         : (N, 32) local observations
      - graph info  : positions, velocities, los_matrix, sinrs, channels
                      (from env info dict)

    The actor/critic process the graph-structured input and produce:
      - actions     : (N,) integer actions per vehicle
    """

    def __init__(
        self,
        state_dim:    int   = 32,
        n_actions:    int   = 16,
        n_agents:     int   = 8,
        d_model:      int   = 128,
        n_heads:      int   = 4,
        n_layers:     int   = 3,
        lr_actor:     float = 3e-5,
        lr_critic:    float = 1e-4,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_eps:     float = 0.1,
        entropy_coef: float = 0.02,
        entropy_min:  float = 0.005,
        value_coef:   float = 0.5,
        epochs:       int   = 4,
        minibatch:    int   = 16,
        grad_clip:    float = 0.5,
        total_episodes: int = 5000,
        device:       str   = "cuda",
    ):
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
        self._episode     = 0
        self._total_eps   = total_episodes
        self.device       = torch.device(
            device if torch.cuda.is_available() else "cpu")

        self.actor  = GraphMAPPOActor(state_dim, n_actions,
                                      d_model, n_heads, n_layers).to(self.device)
        self.critic = GraphMAPPOCritic(state_dim, d_model,
                                       n_heads, n_layers).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),
                                     lr=lr_actor,  eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=lr_critic, eps=1e-5)
        self.actor_sched  = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt,  T_max=total_episodes, eta_min=lr_actor  * 0.1)
        self.critic_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_opt, T_max=total_episodes, eta_min=lr_critic * 0.1)

        self.buffer = GraphRolloutBuffer()
        # Temperature for act_eval: 1.0 = pure argmax, 0.3-0.5 = diversity sampling.
        # Set lower when deterministic argmax causes all agents to pick the same channel.
        self.eval_temperature = 1.0

    def set_eval_temperature(self, t: float):
        """Set evaluation temperature. Use 0.3-0.5 to prevent channel-collapse."""
        self.eval_temperature = float(t)

    def _entropy_coef(self):
        frac = min(1.0, self._episode / max(1, self._total_eps))
        return max(self.entropy_min,
                   self.entropy_coef * (1 - frac) + self.entropy_min * frac)

    # ── Graph tensor helper ────────────────────────────────────────────────

    def _graph_tensors(self, info: dict):
        """Build graph tensors from env info dict."""
        return graph_from_info(info, device=str(self.device))

    # ── Inference ─────────────────────────────────────────────────────────

    def _batched_graph(self, info: dict):
        """Single graph with a leading batch dim of 1 (for per-step inference)."""
        ac, ec, af, ef = self._graph_tensors(info)
        return (ac.unsqueeze(0), ec.unsqueeze(0),
                af.unsqueeze(0), ef.unsqueeze(0))

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Stochastic action for training rollout."""
        s = torch.FloatTensor(obs).unsqueeze(0).to(self.device)   # (1,N,S)
        ac, ec, af, ef = self._batched_graph(info)
        dist = self.actor.get_dist(s, ac, ec, af, ef)             # (1,N,A)
        return dist.sample().squeeze(0).cpu().numpy()             # (N,)

    @torch.no_grad()
    def act_eval(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Deterministic greedy action for evaluation."""
        s = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        ac, ec, af, ef = self._batched_graph(info)
        dist = self.actor.get_dist(s, ac, ec, af, ef)
        if self.eval_temperature == 1.0:
            return dist.logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        # Temperature < 1: sharpen logits but preserve inter-agent diversity.
        # Pure argmax collapses all agents to same channel when their graph
        # embeddings are correlated (same street → same CSI → same argmax).
        scaled = dist.logits / self.eval_temperature
        return torch.distributions.Categorical(logits=scaled).sample().squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_with_info(self, obs: np.ndarray, info: dict):
        """Returns (actions, log_probs, value) for rollout storage."""
        s = torch.FloatTensor(obs).unsqueeze(0).to(self.device)   # (1,N,S)
        ac, ec, af, ef = self._batched_graph(info)
        dist      = self.actor.get_dist(s, ac, ec, af, ef)        # (1,N,A)
        actions   = dist.sample()                                  # (1,N)
        log_probs = dist.log_prob(actions)                         # (1,N)
        value     = self.critic(s, ac, ec, af, ef)                 # (1,)
        return (actions.squeeze(0).cpu().numpy(),
                log_probs.squeeze(0).cpu().numpy(),
                float(value.item()))

    # ── Store ──────────────────────────────────────────────────────────────

    def store(self, obs, actions, log_probs, rewards, value, done, info):
        adj_c, eft_c, adj_f, eft_f = self._graph_tensors(info)
        self.buffer.push(
            obs.copy(), actions.copy(), log_probs.copy(),
            float(np.mean(rewards)), float(value), float(done),
            adj_c.cpu(), eft_c.cpu(), adj_f.cpu(), eft_f.cpu()
        )

    # ── Train (called at episode end) ──────────────────────────────────────

    def train(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {}
        self._episode += 1

        T   = len(self.buffer)
        obs = np.array(self.buffer.obs)             # (T, N, S)
        act = np.array(self.buffer.actions)         # (T, N)
        olp = np.array(self.buffer.log_probs)       # (T, N)
        rew = np.array(self.buffer.rewards)         # (T,)
        val = np.array(self.buffer.values)          # (T,)
        don = np.array(self.buffer.dones)           # (T,)

        # Stack graph tensors along time dimension
        adj_c = torch.stack(self.buffer.adj_cur)   # (T, N, N)
        eft_c = torch.stack(self.buffer.eft_cur)   # (T, N, N, E)
        adj_f = torch.stack(self.buffer.adj_fut)
        eft_f = torch.stack(self.buffer.eft_fut)

        # Per-episode reward normalisation
        if rew.std() > 1e-8:
            rew = (rew - rew.mean()) / (rew.std() + 1e-8)

        # GAE
        adv = np.zeros(T, np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv    = val[t + 1] if t + 1 < T else 0.0
            delta = rew[t] + self.gamma * nv * (1 - don[t]) - val[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - don[t]) * gae
            adv[t] = gae
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dev = self.device
        # Each timestep is one graph -> stack along batch dim. (T = batch size)
        obs_t = torch.FloatTensor(obs).to(dev)                   # (T, N, S)
        act_t = torch.LongTensor(act).to(dev)                    # (T, N)
        olp_t = torch.FloatTensor(olp).to(dev)                   # (T, N)
        adv_t = torch.FloatTensor(adv).to(dev)                   # (T,)
        ret_t = torch.FloatTensor(ret).to(dev)                   # (T,)
        val_t = torch.FloatTensor(val).to(dev)                   # (T,)
        adj_c = adj_c.to(dev); eft_c = eft_c.to(dev)
        adj_f = adj_f.to(dev); eft_f = eft_f.to(dev)

        ent_c = self._entropy_coef()
        logs  = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_upd = 0

        for _ in range(self.epochs):
            perm = torch.randperm(T, device=dev)
            for start in range(0, T, self.minibatch):
                idx = perm[start:start + self.minibatch]        # (mb,) timestep indices

                # ── Actor: one batched forward over the minibatch ─────────
                logits  = self.actor(obs_t[idx], adj_c[idx], eft_c[idx],
                                     adj_f[idx], eft_f[idx])     # (mb, N, A)
                dist    = torch.distributions.Categorical(logits=logits)
                new_lp  = dist.log_prob(act_t[idx])              # (mb, N)
                entropy = dist.entropy().mean()
                ratio   = (new_lp - olp_t[idx]).exp()            # (mb, N)
                adv_mb  = adv_t[idx].unsqueeze(1)                # (mb, 1) broadcast over agents
                s1 = ratio * adv_mb
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                actor_loss = -torch.min(s1, s2).mean() - ent_c * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_opt.step()

                # ── Critic: one batched forward, PPO value clipping ───────
                vp    = self.critic(obs_t[idx], adj_c[idx], eft_c[idx],
                                    adj_f[idx], eft_f[idx])      # (mb,)
                old_v = val_t[idx]
                vc    = old_v + (vp - old_v).clamp(-self.clip_eps, self.clip_eps)
                crit_loss = torch.max(F.mse_loss(vp, ret_t[idx]),
                                      F.mse_loss(vc, ret_t[idx]))

                self.critic_opt.zero_grad()
                crit_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_opt.step()

                logs["actor_loss"]  += actor_loss.item()
                logs["critic_loss"] += crit_loss.item()
                logs["entropy"]     += entropy.item()
                n_upd += 1

        # Normalise logs to per-update averages (interpretable loss curves)
        if n_upd > 0:
            for k in logs:
                logs[k] /= n_upd

        self.actor_sched.step()
        self.critic_sched.step()
        self.buffer.clear()
        return logs

    def param_count(self):
        return (sum(p.numel() for p in self.actor.parameters()  if p.requires_grad) +
                sum(p.numel() for p in self.critic.parameters() if p.requires_grad))
