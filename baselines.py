"""Baseline policies for comparison."""

import numpy as np


def random_policy(env, state: np.ndarray) -> np.ndarray:
    return np.random.randint(0, env.n_actions, size=env.n_v2v)


def greedy_csi_policy(env, state: np.ndarray) -> np.ndarray:
    """Best CSI channel + highest power level."""
    high_power_idx = env.n_power - 1
    actions = []
    for i in range(env.n_v2v):
        best_ch = int(np.argmax(env.csi[i]))
        actions.append(high_power_idx * env.n_subchannels + best_ch)
    return np.array(actions, dtype=np.int64)


def round_robin_policy(env, state: np.ndarray, step: int = 0) -> np.ndarray:
    """Assigns channels in round-robin, medium power."""
    mid_power = (env.n_power - 1) // 2
    actions = []
    for i in range(env.n_v2v):
        ch = (i + step) % env.n_subchannels
        actions.append(mid_power * env.n_subchannels + ch)
    return np.array(actions, dtype=np.int64)
