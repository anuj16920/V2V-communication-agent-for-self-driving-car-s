"""
Prioritized Experience Replay (PER)
Implements the Sum-Tree data structure for O(log N) sampling.
Reference: Schaul et al., 2015 "Prioritized Experience Replay"
"""

import numpy as np
from typing import Tuple


class SumTree:
    """Binary sum tree for efficient priority-based sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.data     = [None] * capacity
        self.ptr      = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data):
        leaf_idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf_idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = 1  # root
        while idx < self.capacity - 1:
            left  = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """
    Shared PER buffer for all agents (parameter sharing setup).

    Stores flat (state, action, reward, next_state, done) tuples.
    IS-weights returned for bias correction.
    """

    def __init__(
        self,
        capacity:  int   = 100_000,
        alpha:     float = 0.6,     # priority exponent
        beta_start:float = 0.4,     # IS weight start
        beta_end:  float = 1.0,     # IS weight end (anneal to 1)
        beta_steps:int   = 100_000,
        epsilon:   float = 1e-5,    # min priority
    ):
        self.tree       = SumTree(capacity)
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self.epsilon    = epsilon
        self.max_priority = 1.0
        self._step      = 0

    @property
    def beta(self) -> float:
        frac = min(1.0, self._step / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority ** self.alpha, (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        self._step += 1
        indices, priorities, transitions = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, pri, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(pri)
            transitions.append(data)

        if len(transitions) == 0:
            return None

        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        probs   = np.clip(probs, 1e-10, None)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            np.array(indices,     dtype=np.int64),
            np.array(weights,     dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, pri in zip(indices, priorities):
            p = float(abs(pri)) + self.epsilon
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), p ** self.alpha)

    def clear(self):
        """Reset the buffer — call on curriculum stage transitions to remove stale transitions."""
        cap = self.tree.capacity
        self.tree = SumTree(cap)
        self.max_priority = 1.0
        self._step = 0

    def __len__(self) -> int:
        return self.tree.size
