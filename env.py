"""
V2V Resource Allocation Environment — Enhanced
- Joint channel + power allocation
- Dynamic mobility with realistic path loss
- Packet queues with deadlines
- V2V + V2I interference
- Multi-objective reward with normalization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def jains_fairness(x: np.ndarray) -> float:
    x = np.array(x, dtype=np.float64)
    s = np.sum(x)
    if s <= 1e-12:
        return 0.0
    return float(s ** 2 / (len(x) * np.sum(x ** 2) + 1e-12))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class V2VEnv(gym.Env):
    """
    Enhanced V2V environment.

    State per vehicle (28 dims):
      [0:4]   CSI per subchannel (log-normalized)
      [4:8]   V2V interference estimate per subchannel
      [8:12]  V2I interference per subchannel
      [12]    Queue length (normalized)
      [13]    Deadline urgency (1/deadline)
      [14]    Speed normalized
      [15]    Position x normalized
      [16]    Position y normalized
      [17:21] Prev channel one-hot
      [21]    Prev power normalized
      [22]    Prev SINR normalized
      [23]    Subchannel contention (normalized)
      [24]    Path loss estimate
      [25]    QoS urgency composite
      [26]    Neighbor density (vehicles in proximity)
      [27]    Episode progress (0→1)
    """

    STATE_DIM   = 28
    SINR_THRESH = 1.0       # threshold for reliability reward

    def __init__(
        self,
        n_v2v: int        = 8,
        n_subchannels: int = 4,
        power_levels       = None,
        episode_len: int   = 50,
        noise_power: float = 0.1,
        v2i_scale: float   = 0.25,
        area_size: float   = 500.0,     # metres
        path_loss_exp: float = 3.5,
        curriculum_stage: int = 3,      # 1=collision, 2=PDR, 3=full
        curriculum_alpha: float = 0.0,  # blend factor: 0=pure stage, 1=pure next stage
        normalize_rewards: bool = False, # running mean/std reward normalization
    ):
        super().__init__()
        if power_levels is None:
            power_levels = [0.1, 0.3, 0.6, 1.0]   # 4 levels now

        self.n_v2v          = n_v2v
        self.n_subchannels  = n_subchannels
        self.power_levels   = np.array(power_levels, dtype=np.float32)
        self.n_power        = len(power_levels)
        self.episode_len    = episode_len
        self.noise_power    = noise_power
        self.v2i_scale      = v2i_scale
        self.area_size      = area_size
        self.path_loss_exp  = path_loss_exp
        self.curriculum_stage = curriculum_stage

        self.curriculum_alpha    = curriculum_alpha
        self.normalize_rewards   = normalize_rewards

        self.n_actions      = self.n_subchannels * self.n_power
        self.action_space   = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.n_v2v, self.STATE_DIM),
            dtype=np.float32
        )

        # Running stats for Welford online reward normalization
        self._rew_mean = 0.0
        self._rew_m2   = 0.0
        self._rew_n    = 0

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Positions in metres
        self.positions = np.random.uniform(0, self.area_size, (self.n_v2v, 2)).astype(np.float32)
        self.speeds    = np.random.uniform(20, 120, self.n_v2v).astype(np.float32)  # km/h
        self.headings  = np.random.uniform(0, 2 * np.pi, self.n_v2v).astype(np.float32)

        # CSI: small-scale fading (Rayleigh) × path loss
        self.csi = self._init_csi()

        self.v2i_interference = np.random.uniform(
            0.0, self.v2i_scale, (self.n_v2v, self.n_subchannels)
        ).astype(np.float32)

        self.prev_channels = np.zeros(self.n_v2v, dtype=np.int64)
        self.prev_powers   = np.zeros(self.n_v2v, dtype=np.float32)
        self.prev_sinr     = np.zeros(self.n_v2v, dtype=np.float32)

        # Queues: random initial load
        self.queue    = np.random.randint(3, 10, self.n_v2v).astype(np.float32)
        self.deadline = np.random.randint(4, 10, self.n_v2v).astype(np.float32)

        self.last_metrics = {}
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Channel model
    # ------------------------------------------------------------------
    def _path_loss(self, dist_m: float) -> float:
        """Free-space path loss model (normalized)."""
        d = max(dist_m, 1.0)
        return 1.0 / (1.0 + (d / 100.0) ** self.path_loss_exp)

    def _init_csi(self) -> np.ndarray:
        """Rayleigh fading × distance-based path loss."""
        csi = np.zeros((self.n_v2v, self.n_subchannels), dtype=np.float32)
        for i in range(self.n_v2v):
            # Nearest neighbour distance as proxy for link distance
            dists = np.linalg.norm(self.positions - self.positions[i], axis=1)
            dists[i] = 1e9
            d = np.min(dists)
            pl = self._path_loss(d)
            rayleigh = np.random.rayleigh(0.5, self.n_subchannels)
            csi[i] = np.clip(pl * rayleigh, 0.05, 2.5)
        return csi

    def _update_channel(self):
        """Jakes-like channel evolution + mobility update."""
        # Mobility
        speed_ms = self.speeds / 3.6  # km/h → m/s
        # Random walk with heading persistence
        self.headings += np.random.normal(0, 0.3, self.n_v2v)
        dx = speed_ms * np.cos(self.headings) * 0.1  # dt = 0.1s
        dy = speed_ms * np.sin(self.headings) * 0.1
        self.positions[:, 0] = np.clip(self.positions[:, 0] + dx, 0, self.area_size)
        self.positions[:, 1] = np.clip(self.positions[:, 1] + dy, 0, self.area_size)

        # Channel: AR(1) + fresh Rayleigh component
        alpha = 0.85
        fresh = self._init_csi()
        delta = np.random.normal(0, 0.05, self.csi.shape).astype(np.float32)
        self.csi = np.clip(alpha * self.csi + (1 - alpha) * fresh + delta, 0.05, 2.5)

        # V2I interference AR(1)
        d_v2i = np.random.normal(0, 0.02, self.v2i_interference.shape).astype(np.float32)
        fresh_v2i = np.random.uniform(0, self.v2i_scale, self.v2i_interference.shape).astype(np.float32)
        self.v2i_interference = np.clip(
            0.85 * self.v2i_interference + 0.15 * fresh_v2i + d_v2i,
            0.0, 0.6
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_v2v, self.STATE_DIM), dtype=np.float32)
        ch_counts = np.bincount(self.prev_channels.astype(int), minlength=self.n_subchannels)
        ep_prog   = self.current_step / self.episode_len

        for i in range(self.n_v2v):
            k = 0

            # CSI log-normalized
            obs[i, k:k+4] = np.log1p(self.csi[i]) / np.log1p(2.5)
            k += 4

            # V2V interference estimate (other agents' last channel choices)
            for ch in range(self.n_subchannels):
                cnt = ch_counts[ch] - (1 if self.prev_channels[i] == ch else 0)
                obs[i, k+ch] = cnt / max(1, self.n_v2v - 1)
            k += 4

            # V2I interference
            obs[i, k:k+4] = np.clip(self.v2i_interference[i] / 0.6, 0, 1)
            k += 4

            # Queue length normalized
            obs[i, k] = np.clip(self.queue[i] / 15.0, 0, 1)
            k += 1

            # Deadline urgency
            obs[i, k] = 1.0 / max(1.0, self.deadline[i])
            k += 1

            # Speed
            obs[i, k] = self.speeds[i] / 120.0
            k += 1

            # Position x, y
            obs[i, k]   = self.positions[i, 0] / self.area_size
            obs[i, k+1] = self.positions[i, 1] / self.area_size
            k += 2

            # Prev channel one-hot
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[int(self.prev_channels[i])] = 1.0
            obs[i, k:k+4] = one_hot
            k += 4

            # Prev power normalized
            obs[i, k] = self.prev_powers[i] / max(self.power_levels)
            k += 1

            # Prev SINR normalized
            obs[i, k] = np.clip(self.prev_sinr[i] / 5.0, 0, 1)
            k += 1

            # Channel contention
            my_ch = int(self.prev_channels[i])
            obs[i, k] = (ch_counts[my_ch] - 1) / max(1, self.n_v2v - 1)
            k += 1

            # Path loss proxy (nearest neighbor distance normalized)
            dists = np.linalg.norm(self.positions - self.positions[i], axis=1)
            dists[i] = 1e9
            obs[i, k] = np.clip(1.0 - np.min(dists) / self.area_size, 0, 1)
            k += 1

            # QoS urgency composite
            obs[i, k] = np.clip(
                (self.queue[i] / 15.0) * (1.0 / max(1.0, self.deadline[i])),
                0, 1
            )
            k += 1

            # Neighbor density (vehicles within 150m)
            dists[i] = 0.0
            obs[i, k] = np.sum(dists < 150.0) / max(1, self.n_v2v - 1)
            k += 1

            # Episode progress
            obs[i, k] = ep_prog
            k += 1

        return obs

    # ------------------------------------------------------------------
    # Curriculum helpers
    # ------------------------------------------------------------------
    def set_curriculum_stage(self, stage: int):
        """Switch curriculum stage and reset normalization statistics."""
        self.curriculum_stage = stage
        self.curriculum_alpha = 0.0
        self._rew_mean = 0.0
        self._rew_m2   = 0.0
        self._rew_n    = 0

    def _compute_stage_reward(
        self,
        stage: int,
        pdrs: np.ndarray,
        throughputs: np.ndarray,
        collisions: np.ndarray,
        sinrs: np.ndarray,
        fairness: float,
        reliability: np.ndarray,
        latency_viols: np.ndarray,
        energy_used: np.ndarray,
    ) -> np.ndarray:
        rewards = np.zeros(self.n_v2v, np.float32)
        if stage == 1:
            for i in range(self.n_v2v):
                rewards[i] = (
                    - 5.0 * collisions[i]
                    + 1.0 * pdrs[i]
                    # SINR shaping: prevents agents choosing low-power (low SINR) to avoid collisions
                    + 0.5 * float(np.clip(sinrs[i] / 5.0, 0.0, 1.0))
                    + 0.3 * fairness
                )
        elif stage == 2:
            for i in range(self.n_v2v):
                rewards[i] = (
                    5.0 * pdrs[i]
                    + 2.0 * throughputs[i]
                    - 3.0 * collisions[i]
                    + 1.5 * fairness
                    + 1.0 * reliability[i]
                )
        else:
            for i in range(self.n_v2v):
                rewards[i] = (
                    4.0 * pdrs[i]
                    + 2.5 * throughputs[i]
                    + 2.0 * fairness
                    + 3.0 * reliability[i]
                    - 3.0 * collisions[i]
                    - 2.5 * latency_viols[i]
                    - 0.5 * energy_used[i]
                )
        return rewards

    def _update_and_normalize(self, rewards: np.ndarray) -> np.ndarray:
        """Welford online update + normalize rewards to unit variance."""
        batch_mean = float(np.mean(rewards))
        self._rew_n += 1
        delta = batch_mean - self._rew_mean
        self._rew_mean += delta / self._rew_n
        delta2 = batch_mean - self._rew_mean
        self._rew_m2 += delta * delta2
        var = self._rew_m2 / max(1, self._rew_n - 1)
        std = float(max(np.sqrt(var), 0.1))
        return np.clip((rewards - self._rew_mean) / std, -5.0, 5.0)

    # ------------------------------------------------------------------
    # Action decode
    # ------------------------------------------------------------------
    def decode_actions(self, actions: np.ndarray):
        channels   = actions % self.n_subchannels
        power_idx  = actions // self.n_subchannels
        powers     = self.power_levels[power_idx]
        return channels, power_idx, powers

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions: np.ndarray):
        self.current_step += 1
        self._update_channel()

        channels, power_idx, powers = self.decode_actions(actions)

        throughputs      = np.zeros(self.n_v2v, np.float32)
        sinrs            = np.zeros(self.n_v2v, np.float32)
        pdrs             = np.zeros(self.n_v2v, np.float32)
        collisions       = np.zeros(self.n_v2v, np.float32)
        latency_viols    = np.zeros(self.n_v2v, np.float32)
        energy_used      = np.zeros(self.n_v2v, np.float32)
        delivered        = np.zeros(self.n_v2v, np.float32)

        for i in range(self.n_v2v):
            ch       = int(channels[i])
            tx_power = float(powers[i])
            signal   = tx_power * self.csi[i, ch]

            v2v_interf  = 0.0
            coll_count  = 0
            for j in range(self.n_v2v):
                if i != j and channels[j] == ch:
                    v2v_interf += 0.35 * float(powers[j]) * float(self.csi[j, ch])
                    coll_count += 1

            v2i_interf  = float(self.v2i_interference[i, ch])
            total_interf = self.noise_power + v2v_interf + v2i_interf
            sinr         = signal / max(1e-8, total_interf)
            rate         = float(np.log2(1.0 + sinr))
            pdr          = sigmoid(8.0 * (sinr - 0.8))
            coll_ratio   = coll_count / max(1, self.n_v2v - 1)
            service      = min(float(self.queue[i]), rate * pdr * 1.5)

            throughputs[i]  = rate
            sinrs[i]        = sinr
            pdrs[i]         = pdr
            collisions[i]   = coll_ratio
            energy_used[i]  = tx_power
            delivered[i]    = service

        # Update queues + deadlines
        for i in range(self.n_v2v):
            arrival = np.random.poisson(1.2)
            self.queue[i] = max(0.0, self.queue[i] - delivered[i]) + arrival
            self.deadline[i] -= 1
            if self.deadline[i] <= 0:
                if self.queue[i] > 0.5:
                    latency_viols[i] = 1.0
                self.deadline[i] = float(np.random.randint(4, 10))

        fairness    = jains_fairness(throughputs)
        reliability = (sinrs >= self.SINR_THRESH).astype(np.float32)

        # Curriculum Learning Reward with smooth alpha-blending between stages
        rw_args = (pdrs, throughputs, collisions, sinrs, fairness,
                   reliability, latency_viols, energy_used)
        alpha = self.curriculum_alpha
        if 0.0 < alpha < 1.0 and self.curriculum_stage < 3:
            r_curr = self._compute_stage_reward(self.curriculum_stage, *rw_args)
            r_next = self._compute_stage_reward(self.curriculum_stage + 1, *rw_args)
            rewards = (1.0 - alpha) * r_curr + alpha * r_next
        else:
            rewards = self._compute_stage_reward(self.curriculum_stage, *rw_args)

        if self.normalize_rewards:
            rewards = self._update_and_normalize(rewards)

        self.prev_channels = channels.copy()
        self.prev_powers   = powers.copy()
        self.prev_sinr     = sinrs.copy()

        self.last_metrics = {
            "avg_pdr":               float(np.mean(pdrs)),
            "avg_throughput":        float(np.mean(throughputs)),
            "avg_collision":         float(np.mean(collisions)),
            "avg_sinr":              float(np.mean(sinrs)),
            "fairness":              float(fairness),
            "avg_energy":            float(np.mean(energy_used)),
            "latency_violation_rate":float(np.mean(latency_viols)),
            "reliability_rate":      float(np.mean(reliability)),
        }

        obs        = self._get_obs()
        terminated = self.current_step >= self.episode_len
        truncated  = False
        return obs, rewards, terminated, truncated, self.last_metrics
