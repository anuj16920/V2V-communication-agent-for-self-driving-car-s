"""
Manhattan Grid V2V Environment

Replaces the random-walk open-space model with a realistic urban scenario:
  - Vehicles constrained to a Manhattan grid of streets
  - LoS / NLoS path loss based on road geometry (buildings block cross-street links)
  - 3GPP-inspired urban V2V channel model
  - Intersection-aware observation features
  - All curriculum, reward, and action interfaces identical to V2VEnv (drop-in replacement)

Grid layout (default 5×5 blocks, 250 m blocks, 15 m streets ≈ 1.34 km × 1.34 km):

   ─── v_streets[0]  v_streets[1] … v_streets[5]
   h_streets[5]  ┼──────────┼──────────┼
                 │          │          │
   h_streets[4]  ┼──────────┼──────────┼
                 │  block   │  block   │
   h_streets[3]  ┼──────────┼──────────┼
                 │          │          │
   …             ┼──────────┼──────────┼
   h_streets[0]  ┼──────────┼──────────┼

Directions:  EAST=0  NORTH=1  WEST=2  SOUTH=3

State per vehicle (32 dims = 28 base + 4 Manhattan):
  [0:4]   CSI per subchannel
  [4:8]   V2V interference estimate
  [8:12]  V2I interference
  [12]    Queue length
  [13]    Deadline urgency
  [14]    Speed
  [15:17] Position x, y
  [17:21] Prev channel one-hot
  [21]    Prev power
  [22]    Prev SINR
  [23]    Subchannel contention
  [24]    Path loss estimate
  [25]    QoS urgency
  [26]    Neighbor density (vehicles within 150 m)
  [27]    Episode progress
  [28]    LoS ratio  (fraction of other vehicles in Line-of-Sight)
  [29]    Distance to nearest intersection  (normalized)
  [30]    Road orientation  (0 = horizontal, 1 = vertical)
  [31]    Heading  (0 = E, 0.25 = N, 0.5 = W, 0.75 = S)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ─── Grid constants ───────────────────────────────────────────────────────────
N_BLOCKS    = 5        # blocks per axis
BLOCK_SIZE  = 250.0    # metres per block
STREET_W    = 15.0     # street half-width radius (centre to kerb)
GRID_STEP   = BLOCK_SIZE + STREET_W   # 265 m between street centrelines
N_STREETS   = N_BLOCKS + 1            # 6 streets per axis

# Street centreline positions (metres)
H_STREETS = np.array([STREET_W / 2.0 + i * GRID_STEP for i in range(N_STREETS)])
V_STREETS = np.array([STREET_W / 2.0 + i * GRID_STEP for i in range(N_STREETS)])

AREA_X = V_STREETS[-1] + STREET_W / 2.0   # ≈ 1340 m
AREA_Y = H_STREETS[-1] + STREET_W / 2.0

INTERSECTION_R = 20.0   # radius considered "at intersection" (metres)
DT             = 0.1    # simulation time step (seconds)

# Direction vectors: EAST=0, NORTH=1, WEST=2, SOUTH=3
DIR_VEC = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
# Turn table: [dir][action]  action: 0=straight, 1=turn-left, 2=turn-right
TURN_TABLE = {
    0: {0: 0, 1: 1, 2: 3},   # EAST:  straight=E, left=N, right=S
    1: {0: 1, 1: 2, 2: 0},   # NORTH: straight=N, left=W, right=E
    2: {0: 2, 1: 3, 2: 1},   # WEST:  straight=W, left=S, right=N
    3: {0: 3, 1: 0, 2: 2},   # SOUTH: straight=S, left=E, right=W
}
TURN_PROBS = [0.50, 0.25, 0.25]    # straight / left / right


def _jains_fairness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    s = np.sum(x)
    if s <= 1e-12:
        return 0.0
    return float(s ** 2 / (len(x) * np.sum(x ** 2) + 1e-12))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ─── Manhattan env ────────────────────────────────────────────────────────────
class V2VManhattanEnv(gym.Env):
    """
    Drop-in replacement for V2VEnv with Manhattan grid mobility and urban channel.
    """

    STATE_DIM   = 32          # 28 base + 4 Manhattan features
    SINR_THRESH = 1.0

    # NLoS attenuation relative to LoS (linear scale ~-15 dB building penetration)
    NLOS_FACTOR = 10 ** (-1.5)   # ≈ 0.0316

    def __init__(
        self,
        n_v2v: int         = 8,
        n_subchannels: int = 4,
        power_levels       = None,
        episode_len: int   = 50,
        noise_power: float = 0.1,
        v2i_scale: float   = 0.20,    # slightly lower V2I in urban (RSUs closer)
        curriculum_stage: int = 3,
        curriculum_alpha: float = 0.0,
        normalize_rewards: bool = False,
    ):
        super().__init__()
        if power_levels is None:
            power_levels = [0.1, 0.3, 0.6, 1.0]

        self.n_v2v          = n_v2v
        self.n_subchannels  = n_subchannels
        self.power_levels   = np.array(power_levels, dtype=np.float32)
        self.n_power        = len(power_levels)
        self.episode_len    = episode_len
        self.noise_power    = noise_power
        self.v2i_scale      = v2i_scale
        self.curriculum_stage = curriculum_stage
        self.curriculum_alpha = curriculum_alpha
        self.normalize_rewards = normalize_rewards

        self.n_actions   = self.n_subchannels * self.n_power
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.n_v2v, self.STATE_DIM),
            dtype=np.float32
        )

        # Reward normalisation (Welford)
        self._rew_mean = 0.0
        self._rew_m2   = 0.0
        self._rew_n    = 0

        self.reset()

    # ── Curriculum helpers (same API as V2VEnv) ──────────────────────────────
    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage
        self.curriculum_alpha = 0.0
        self._rew_mean = 0.0
        self._rew_m2   = 0.0
        self._rew_n    = 0

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Place vehicles on random streets
        self.positions  = np.zeros((self.n_v2v, 2), dtype=np.float32)
        self.directions = np.zeros(self.n_v2v, dtype=np.int32)   # 0-3
        self.speeds     = np.random.uniform(20, 60, self.n_v2v).astype(np.float32)  # km/h

        for i in range(self.n_v2v):
            x, y, d = self._random_street_position()
            self.positions[i] = [x, y]
            self.directions[i] = d

        # Channel init
        self.csi = self._init_csi()
        self.v2i_interference = np.random.uniform(
            0.0, self.v2i_scale, (self.n_v2v, self.n_subchannels)
        ).astype(np.float32)

        # History
        self.prev_channels = np.zeros(self.n_v2v, dtype=np.int64)
        self.prev_powers   = np.zeros(self.n_v2v, dtype=np.float32)
        self.prev_sinr     = np.zeros(self.n_v2v, dtype=np.float32)

        # Queues
        self.queue    = np.random.randint(3, 10, self.n_v2v).astype(np.float32)
        self.deadline = np.random.randint(4, 10, self.n_v2v).astype(np.float32)

        return self._get_obs(), {}

    # ── Street/grid helpers ──────────────────────────────────────────────────
    def _random_street_position(self):
        """Return (x, y, direction) on a random street."""
        if np.random.random() < 0.5:
            # Horizontal street
            hi = np.random.randint(0, N_STREETS)
            y  = H_STREETS[hi]
            x  = np.random.uniform(V_STREETS[0], V_STREETS[-1])
            d  = np.random.choice([0, 2])   # EAST or WEST
        else:
            # Vertical street
            vi = np.random.randint(0, N_STREETS)
            x  = V_STREETS[vi]
            y  = np.random.uniform(H_STREETS[0], H_STREETS[-1])
            d  = np.random.choice([1, 3])   # NORTH or SOUTH
        return float(x), float(y), int(d)

    def _snap_to_street(self, x: float, y: float, direction: int):
        """Snap position to the nearest street matching the direction."""
        if direction in (0, 2):  # EAST/WEST → horizontal street
            nearest_hi = int(np.argmin(np.abs(H_STREETS - y)))
            return x, float(H_STREETS[nearest_hi])
        else:                    # NORTH/SOUTH → vertical street
            nearest_vi = int(np.argmin(np.abs(V_STREETS - x)))
            return float(V_STREETS[nearest_vi]), y

    def _nearest_intersection_dist(self, x: float, y: float) -> float:
        """Distance from (x,y) to the nearest grid intersection."""
        best = np.inf
        for vx in V_STREETS:
            for hy in H_STREETS:
                d = (x - vx) ** 2 + (y - hy) ** 2
                if d < best:
                    best = d
        return float(np.sqrt(best))

    def _on_street(self, x: float, y: float):
        """
        Returns (on_h_street: bool, h_idx: int, on_v_street: bool, v_idx: int).
        A position is "on" a street if it is within STREET_W/2 of a centreline.
        """
        h_dists = np.abs(H_STREETS - y)
        v_dists = np.abs(V_STREETS - x)
        h_idx   = int(np.argmin(h_dists))
        v_idx   = int(np.argmin(v_dists))
        on_h    = h_dists[h_idx] < STREET_W / 2.0
        on_v    = v_dists[v_idx] < STREET_W / 2.0
        return on_h, h_idx, on_v, v_idx

    # ── Mobility ─────────────────────────────────────────────────────────────
    def _update_mobility(self):
        """Move all vehicles along Manhattan streets for one DT second."""
        for i in range(self.n_v2v):
            speed_ms = self.speeds[i] / 3.6  # km/h → m/s
            dx, dy   = DIR_VEC[self.directions[i]]
            new_x    = self.positions[i, 0] + dx * speed_ms * DT
            new_y    = self.positions[i, 1] + dy * speed_ms * DT

            # Check if passing through an intersection
            crossed, ix, iy = self._crossed_intersection(
                self.positions[i, 0], self.positions[i, 1],
                new_x, new_y, self.directions[i]
            )
            if crossed:
                # Snap to intersection, choose new direction
                new_d = TURN_TABLE[self.directions[i]][
                    np.random.choice([0, 1, 2], p=TURN_PROBS)
                ]
                self.directions[i] = new_d
                # Continue remainder of step in new direction
                new_x, new_y = ix, iy

            # Boundary reflection: if vehicle goes off grid, reverse direction
            if new_x < V_STREETS[0] or new_x > V_STREETS[-1]:
                self.directions[i] = 2 if self.directions[i] == 0 else 0
                new_x = np.clip(new_x, V_STREETS[0], V_STREETS[-1])
            if new_y < H_STREETS[0] or new_y > H_STREETS[-1]:
                self.directions[i] = 3 if self.directions[i] == 1 else 1
                new_y = np.clip(new_y, H_STREETS[0], H_STREETS[-1])

            # Snap to street (keeps vehicle on road centreline)
            new_x, new_y = self._snap_to_street(new_x, new_y, self.directions[i])
            self.positions[i] = [new_x, new_y]

        # Slight speed variation (urban stop-and-go)
        self.speeds += np.random.normal(0, 2, self.n_v2v)
        self.speeds  = np.clip(self.speeds, 10, 60).astype(np.float32)

    def _crossed_intersection(self, x0, y0, x1, y1, direction):
        """
        Check if the step (x0,y0)→(x1,y1) crosses a grid intersection.
        Returns (crossed: bool, int_x: float, int_y: float).
        """
        if direction == 0:   # EAST: x increasing, y ~ const
            for vx in V_STREETS:
                if x0 < vx <= x1:
                    return True, vx, y0
        elif direction == 1: # NORTH: y increasing, x ~ const
            for hy in H_STREETS:
                if y0 < hy <= y1:
                    return True, x0, hy
        elif direction == 2: # WEST: x decreasing
            for vx in reversed(V_STREETS):
                if x1 <= vx < x0:
                    return True, vx, y0
        elif direction == 3: # SOUTH: y decreasing
            for hy in reversed(H_STREETS):
                if y1 <= hy < y0:
                    return True, x0, hy
        return False, x0, y0

    # ── Channel model ─────────────────────────────────────────────────────────
    def _is_los(self, i: int, j: int) -> bool:
        """
        LoS if both vehicles are on the same street segment.
        - Same horizontal street AND between the same pair of cross-streets
        - Same vertical street AND between the same pair of cross-streets
        Distance cap: LoS degrades at >500 m even on straight road.
        """
        xi, yi = self.positions[i]
        xj, yj = self.positions[j]

        dist = float(np.hypot(xi - xj, yi - yj))
        if dist > 500.0:   # beyond LoS horizon in urban env
            return False

        # Within intersection radius → definitely LoS
        if dist < INTERSECTION_R:
            return True

        on_h_i, hi_i, on_v_i, vi_i = self._on_street(xi, yi)
        on_h_j, hi_j, on_v_j, vi_j = self._on_street(xj, yj)

        # Same horizontal street
        if on_h_i and on_h_j and hi_i == hi_j:
            return True
        # Same vertical street
        if on_v_i and on_v_j and vi_i == vi_j:
            return True
        return False

    def _path_loss(self, dist_m: float, is_los: bool) -> float:
        """
        Urban V2V path loss (linear scale, normalised so PL(10m)≈1).
        LoS: exponent 2.1 (near free-space along open street).
        NLoS: exponent 3.8 + building penetration loss (~-15 dB).
        """
        d = max(dist_m, 1.0)
        if is_los:
            return 1.0 / (1.0 + (d / 100.0) ** 2.1)
        else:
            return self.NLOS_FACTOR / (1.0 + (d / 100.0) ** 3.8)

    def _init_csi(self) -> np.ndarray:
        """
        Rayleigh small-scale fading × urban path loss.
        Uses LoS/NLoS to nearest neighbour for initial CSI.
        """
        csi = np.zeros((self.n_v2v, self.n_subchannels), dtype=np.float32)
        for i in range(self.n_v2v):
            xi, yi = self.positions[i]
            # Find nearest other vehicle
            dists = [np.hypot(xi - self.positions[j, 0], yi - self.positions[j, 1])
                     if j != i else 1e9 for j in range(self.n_v2v)]
            j_near = int(np.argmin(dists))
            d_near = float(dists[j_near])
            los    = self._is_los(i, j_near)
            pl     = self._path_loss(d_near, los)
            rayleigh = np.random.rayleigh(0.5, self.n_subchannels)
            csi[i]   = np.clip(pl * rayleigh, 0.02, 3.0)
        return csi

    def _update_channel(self):
        """AR(1) Jakes-like channel evolution + mobility step."""
        self._update_mobility()

        alpha = 0.85
        fresh = self._init_csi()
        delta = np.random.normal(0, 0.04, self.csi.shape).astype(np.float32)
        self.csi = np.clip(alpha * self.csi + (1 - alpha) * fresh + delta, 0.02, 3.0)

        # V2I interference AR(1)
        fresh_v2i = np.random.uniform(0, self.v2i_scale, self.v2i_interference.shape).astype(np.float32)
        dv2i      = np.random.normal(0, 0.02, self.v2i_interference.shape).astype(np.float32)
        self.v2i_interference = np.clip(
            0.85 * self.v2i_interference + 0.15 * fresh_v2i + dv2i, 0.0, 0.5
        )

    # ── Observation (32-dim) ─────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        obs       = np.zeros((self.n_v2v, self.STATE_DIM), dtype=np.float32)
        ch_counts = np.bincount(self.prev_channels.astype(int), minlength=self.n_subchannels)
        ep_prog   = self.current_step / self.episode_len

        # Precompute LoS matrix
        los_matrix = np.zeros((self.n_v2v, self.n_v2v), dtype=bool)
        for i in range(self.n_v2v):
            for j in range(i + 1, self.n_v2v):
                los_matrix[i, j] = los_matrix[j, i] = self._is_los(i, j)

        for i in range(self.n_v2v):
            xi, yi = self.positions[i]
            k = 0

            # [0:4] CSI
            obs[i, k:k+4] = np.log1p(self.csi[i]) / np.log1p(3.0)
            k += 4

            # [4:8] V2V interference estimate
            for ch in range(self.n_subchannels):
                cnt = ch_counts[ch] - (1 if self.prev_channels[i] == ch else 0)
                obs[i, k + ch] = cnt / max(1, self.n_v2v - 1)
            k += 4

            # [8:12] V2I interference
            obs[i, k:k+4] = np.clip(self.v2i_interference[i] / 0.5, 0, 1)
            k += 4

            # [12] Queue length
            obs[i, k] = np.clip(self.queue[i] / 15.0, 0, 1)
            k += 1

            # [13] Deadline urgency
            obs[i, k] = 1.0 / max(1.0, self.deadline[i])
            k += 1

            # [14] Speed (urban: 0-60 km/h)
            obs[i, k] = self.speeds[i] / 60.0
            k += 1

            # [15:17] Position x, y
            obs[i, k]   = xi / AREA_X
            obs[i, k+1] = yi / AREA_Y
            k += 2

            # [17:21] Prev channel one-hot
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[int(self.prev_channels[i])] = 1.0
            obs[i, k:k+4] = one_hot
            k += 4

            # [21] Prev power
            obs[i, k] = self.prev_powers[i] / max(self.power_levels)
            k += 1

            # [22] Prev SINR
            obs[i, k] = np.clip(self.prev_sinr[i] / 5.0, 0, 1)
            k += 1

            # [23] Subchannel contention
            my_ch = int(self.prev_channels[i])
            obs[i, k] = (ch_counts[my_ch] - 1) / max(1, self.n_v2v - 1)
            k += 1

            # [24] Path loss proxy (nearest neighbour, LoS-aware)
            dists = [np.hypot(xi - self.positions[j, 0], yi - self.positions[j, 1])
                     if j != i else 1e9 for j in range(self.n_v2v)]
            j_near = int(np.argmin(dists))
            d_near = float(dists[j_near])
            los_near = los_matrix[i, j_near]
            obs[i, k] = self._path_loss(d_near, los_near)
            k += 1

            # [25] QoS urgency composite
            obs[i, k] = np.clip(
                (self.queue[i] / 15.0) * (1.0 / max(1.0, self.deadline[i])), 0, 1
            )
            k += 1

            # [26] Neighbor density (within 150 m)
            close = sum(
                1 for j in range(self.n_v2v)
                if j != i and float(dists[j]) < 150.0
            )
            obs[i, k] = close / max(1, self.n_v2v - 1)
            k += 1

            # [27] Episode progress
            obs[i, k] = ep_prog
            k += 1

            # ── Manhattan-specific features ──────────────────────────
            # [28] LoS ratio
            n_los = int(np.sum(los_matrix[i]))
            obs[i, k] = n_los / max(1, self.n_v2v - 1)
            k += 1

            # [29] Distance to nearest intersection (normalised by 265 m grid step)
            obs[i, k] = np.clip(
                self._nearest_intersection_dist(xi, yi) / GRID_STEP, 0, 1
            )
            k += 1

            # [30] Road orientation (0 = horizontal, 1 = vertical)
            obs[i, k] = 1.0 if self.directions[i] in (1, 3) else 0.0
            k += 1

            # [31] Heading (0=E, 0.25=N, 0.5=W, 0.75=S)
            obs[i, k] = self.directions[i] / 4.0
            k += 1

        return obs

    # ── Action decode ────────────────────────────────────────────────────────
    def decode_actions(self, actions: np.ndarray):
        channels  = actions % self.n_subchannels
        power_idx = actions // self.n_subchannels
        powers    = self.power_levels[power_idx]
        return channels, power_idx, powers

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        self.current_step += 1
        self._update_channel()   # mobility + channel update

        channels, power_idx, powers = self.decode_actions(actions)

        throughputs   = np.zeros(self.n_v2v, np.float32)
        sinrs         = np.zeros(self.n_v2v, np.float32)
        pdrs          = np.zeros(self.n_v2v, np.float32)
        collisions    = np.zeros(self.n_v2v, np.float32)
        solo_bonuses  = np.zeros(self.n_v2v, np.float32)  # 1/(1+co-channel count)
        latency_viols = np.zeros(self.n_v2v, np.float32)
        energy_used   = np.zeros(self.n_v2v, np.float32)
        delivered     = np.zeros(self.n_v2v, np.float32)

        # LoS matrix for interference calculation
        los_matrix = np.zeros((self.n_v2v, self.n_v2v), dtype=bool)
        for i in range(self.n_v2v):
            for j in range(i + 1, self.n_v2v):
                los_matrix[i, j] = los_matrix[j, i] = self._is_los(i, j)

        for i in range(self.n_v2v):
            ch       = int(channels[i])
            tx_power = float(powers[i])
            signal   = tx_power * self.csi[i, ch]

            v2v_interf  = 0.0
            coll_count  = 0
            for j in range(self.n_v2v):
                if j != i and channels[j] == ch:
                    # LoS interference is full strength; NLoS attenuated by NLOS_FACTOR
                    los_factor = 1.0 if los_matrix[i, j] else self.NLOS_FACTOR
                    v2v_interf += los_factor * 0.35 * float(powers[j]) * float(self.csi[j, ch])
                    coll_count += 1

            v2i_interf   = float(self.v2i_interference[i, ch])
            total_interf = self.noise_power + v2v_interf + v2i_interf
            sinr         = signal / max(1e-8, total_interf)
            rate         = float(np.log2(1.0 + sinr))
            pdr          = _sigmoid(8.0 * (sinr - 0.8))
            coll_ratio   = coll_count / max(1, self.n_v2v - 1)
            service      = min(float(self.queue[i]), rate * pdr * 1.5)

            throughputs[i]  = rate
            sinrs[i]        = sinr
            pdrs[i]         = pdr
            collisions[i]   = coll_ratio
            solo_bonuses[i] = 1.0 / (1.0 + coll_count)  # 1.0 alone, 0.5 with 1 sharer, ...
            energy_used[i] = tx_power
            delivered[i]   = service

        # Queue + deadline update
        for i in range(self.n_v2v):
            arrival = np.random.poisson(1.2)
            self.queue[i] = max(0.0, self.queue[i] - delivered[i]) + arrival
            self.deadline[i] -= 1
            if self.deadline[i] <= 0:
                if self.queue[i] > 0.5:
                    latency_viols[i] = 1.0
                self.deadline[i] = float(np.random.randint(4, 10))

        fairness    = _jains_fairness(throughputs)
        reliability = (sinrs >= self.SINR_THRESH).astype(np.float32)

        # ── Intersection safety bonus ────────────────────────────────
        # Extra reward for delivering packets near intersections (safety-critical)
        int_bonus = np.array([
            0.5 * float(self._nearest_intersection_dist(
                self.positions[i, 0], self.positions[i, 1]
            ) < INTERSECTION_R) * pdrs[i]
            for i in range(self.n_v2v)
        ], dtype=np.float32)

        # ── Reward (shared curriculum API) ───────────────────────────
        rw_args = (pdrs, throughputs, collisions, sinrs, fairness,
                   reliability, latency_viols, energy_used, int_bonus, solo_bonuses)
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

        los_ratio_avg = float(np.mean([
            np.sum(los_matrix[i]) / max(1, self.n_v2v - 1)
            for i in range(self.n_v2v)
        ]))

        info = {
            "avg_pdr":               float(np.mean(pdrs)),
            "avg_throughput":        float(np.mean(throughputs)),
            "avg_collision":         float(np.mean(collisions)),
            "avg_sinr":              float(np.mean(sinrs)),
            "fairness":              float(fairness),
            "avg_energy":            float(np.mean(energy_used)),
            "latency_violation_rate":float(np.mean(latency_viols)),
            "reliability_rate":      float(np.mean(reliability)),
            "avg_los_ratio":         los_ratio_avg,
            "avg_int_bonus":         float(np.mean(int_bonus)),
        }

        obs        = self._get_obs()
        terminated = self.current_step >= self.episode_len
        return obs, rewards, terminated, False, info

    # ── Reward functions (same 3-stage curriculum as V2VEnv) ─────────────────
    def _compute_stage_reward(
        self,
        stage: int,
        pdrs, throughputs, collisions, sinrs, fairness,
        reliability, latency_viols, energy_used, int_bonus, solo_bonuses,
    ) -> np.ndarray:
        rewards = np.zeros(self.n_v2v, np.float32)
        if stage == 1:
            for i in range(self.n_v2v):
                rewards[i] = (
                    - 5.0 * collisions[i]
                    + 3.0 * solo_bonuses[i]   # 3.0 alone → 1.5 with 1 sharer → breaks greedy
                    + 0.5 * pdrs[i]
                    + 0.3 * float(np.clip(sinrs[i] / 5.0, 0.0, 1.0))
                    + 0.2 * fairness
                )
        elif stage == 2:
            for i in range(self.n_v2v):
                rewards[i] = (
                    5.0 * pdrs[i]
                    + 2.0 * throughputs[i]
                    - 3.0 * collisions[i]
                    + 1.5 * solo_bonuses[i]   # KEEP channel-diversity incentive past stage 1
                    + 1.5 * fairness
                    + 1.0 * reliability[i]
                    + 1.0 * int_bonus[i]   # intersection safety bonus from stage 2
                )
        else:
            for i in range(self.n_v2v):
                rewards[i] = (
                    4.0 * pdrs[i]
                    + 2.5 * throughputs[i]
                    + 2.0 * fairness
                    + 3.0 * reliability[i]
                    - 3.0 * collisions[i]
                    + 1.0 * solo_bonuses[i]   # KEEP channel-diversity incentive in full objective
                    - 2.5 * latency_viols[i]
                    - 0.5 * energy_used[i]
                    + 1.5 * int_bonus[i]   # safety-critical intersection delivery
                )
        return rewards

    def _update_and_normalize(self, rewards: np.ndarray) -> np.ndarray:
        batch_mean = float(np.mean(rewards))
        self._rew_n  += 1
        delta         = batch_mean - self._rew_mean
        self._rew_mean += delta / self._rew_n
        delta2         = batch_mean - self._rew_mean
        self._rew_m2  += delta * delta2
        var   = self._rew_m2 / max(1, self._rew_n - 1)
        std   = float(max(np.sqrt(var), 0.1))
        return np.clip((rewards - self._rew_mean) / std, -5.0, 5.0)
