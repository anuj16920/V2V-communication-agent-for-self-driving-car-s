"""
SOTA Training Script — V2V Multi-Agent Resource Allocation

Key improvements over train_improved.py:
  1. Adaptive curriculum: stage advances on rolling metric threshold, not fixed episode count
  2. PER buffer cleared on every curriculum transition (removes stale reward signal)
  3. LR warm-restart at each transition (fresh optimisation landscape)
  4. Smooth alpha-blending across stage boundary (optional, default off)
  5. Reward normalization per stage (consistent value function scale)
  6. QMIX: mixing_embed=64, d_model=256, tau=0.005, polynomial ε-decay
  7. MAPPO with curriculum (same 3 stages as QMIX)
  8. Value-function clipping + entropy decay in MAPPO
  9. Best-model checkpointing every evaluation window
  10. Transfer learning: n=16,24 init from n=8 trained weights
"""

import os
import time
import copy
import numpy as np
import torch
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from env import V2VEnv
from env_manhattan import V2VManhattanEnv
from agent_qmix import QMIXAgent
from agent_mappo import MAPPOAgent
from baselines import random_policy, greedy_csi_policy, round_robin_policy

# ─── Scenario selector ──────────────────────────────────────────────────────
USE_MANHATTAN = True    # False = original random-walk env, True = Manhattan grid

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

os.makedirs("results",           exist_ok=True)
os.makedirs("models_sota",       exist_ok=True)

# ─── Curriculum advancement thresholds ──────────────────────────────────────
# Stage 1 → 2 when rolling collision rate < threshold  (agents learned to spread)
# Stage 2 → 3 when rolling PDR > threshold             (agents learned to deliver)
# Manhattan: 8 vehicles × 4 channels → perfect spread = 1/7 ≈ 0.143 per vehicle.
# Both agents consistently land at ~0.24-0.25 collision. Threshold 0.28 lets them
# advance organically into Stage 2/3 rather than always hitting MAX_STAGE_EPS.
# Random-walk threshold was 0.15 (easier open-space coordination).
_COLL_THRESH = 0.28 if USE_MANHATTAN else 0.15
# Stage 2 PDR threshold lowered 0.40 → 0.25 for Manhattan: NLoS building loss caps
# achievable PDR far below open-space; 0.40 was never met organically (plateaued ~0.21).
_PDR_THRESH = 0.25 if USE_MANHATTAN else 0.40
STAGE_ADVANCE = {
    1: dict(metric="collision", op="lt", threshold=_COLL_THRESH),
    2: dict(metric="pdr",       op="gt", threshold=_PDR_THRESH),
}
ROLLING_WINDOW  = 100   # episodes to average over
MIN_STAGE_EPS   = 600   # never advance before this many episodes in current stage
MAX_STAGE_EPS   = 2000  # force advance if stuck this long
EVAL_INTERVAL   = 500   # episodes between periodic evaluations


# ─── Utilities ───────────────────────────────────────────────────────────────
def rolling_mean(history: deque) -> float:
    return float(np.mean(list(history))) if len(history) > 0 else 0.0


def should_advance(history: deque, stage: int, stage_eps: int) -> bool:
    if stage not in STAGE_ADVANCE or stage_eps < MIN_STAGE_EPS:
        return False
    if len(history) < ROLLING_WINDOW:
        return False
    cfg = STAGE_ADVANCE[stage]
    avg = rolling_mean(history)
    return (avg < cfg["threshold"]) if cfg["op"] == "lt" else (avg > cfg["threshold"])


def evaluate_policy(env, policy_fn, episodes: int = 30) -> dict:
    keys = ["avg_pdr", "avg_throughput", "avg_collision", "avg_sinr",
            "fairness", "avg_energy", "latency_violation_rate", "reliability_rate"]
    acc     = defaultdict(list)
    rew_acc = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_rew   = 0.0
        while True:
            actions            = policy_fn(env, state)
            state, rews, done, _, info = env.step(actions)
            ep_rew += float(np.mean(rews))
            if done:
                break
        rew_acc.append(ep_rew)
        for k in keys:
            acc[k].append(info[k])

    result = {"Reward": float(np.mean(rew_acc))}
    label_map = {
        "avg_pdr": "PDR", "avg_throughput": "Throughput",
        "avg_collision": "Collision", "avg_sinr": "SINR",
        "fairness": "Fairness", "avg_energy": "Energy",
        "latency_violation_rate": "LatencyViolation",
        "reliability_rate": "Reliability",
    }
    for k, lbl in label_map.items():
        result[lbl] = float(np.mean(acc[k]))
    return result


# ─── QMIX SOTA Training ──────────────────────────────────────────────────────
def train_qmix_sota(
    n_v2v:          int   = 8,
    total_episodes: int   = 4000,
    transfer_from:  dict  = None,   # state_dicts from a smaller-density run
) -> dict:
    """
    QMIX with adaptive 3-stage curriculum and all SOTA fixes.
    """
    scenario = "Manhattan Grid" if USE_MANHATTAN else "Random Walk"
    print(f"\n{'='*60}")
    print(f"  [QMIX SOTA | {scenario}] n_v2v={n_v2v}  eps={total_episodes}")
    print(f"{'='*60}")

    LR     = 1e-4
    D_MODEL= 256   # up from 128
    N_HEAD = 8     # up from 4

    EnvClass = V2VManhattanEnv if USE_MANHATTAN else V2VEnv
    env = EnvClass(
        n_v2v=n_v2v, n_subchannels=4, episode_len=50,
        curriculum_stage=1,
        normalize_rewards=True,
    )

    agent = QMIXAgent(
        state_dim        = env.STATE_DIM,
        n_actions        = env.n_actions,
        n_agents         = n_v2v,
        d_model          = D_MODEL,
        n_heads          = N_HEAD,
        n_layers         = 3,
        d_ff             = 512,
        dropout          = 0.1,
        mixing_embed_dim = 64,     # was 32, more expressive hypernetwork
        lr               = LR,
        gamma            = 0.99,
        batch_size       = 256,
        n_steps          = 3,
        tau              = 0.005,  # was 0.001, faster target adaptation
        grad_clip        = 10.0,
        update_freq      = 4,
        min_replay_size  = 3000,   # warm-up before training begins
        per_capacity     = 200_000,
        epsilon_start    = 1.0,
        epsilon_end      = 0.05,
        epsilon_steps    = 120_000, # more exploration for urban coordination
        device           = DEVICE,
    )
    print(f"  QMIX param count: {agent.param_count():,}")

    # Optional transfer learning from smaller density
    if transfer_from is not None:
        try:
            agent.q_network.load_state_dict(transfer_from["q_network"])
            agent.target_q_network.load_state_dict(transfer_from["q_network"])
            print(f"  Transferred Q-network weights from n={transfer_from.get('n_v2v','?')}")
        except Exception as e:
            print(f"  Transfer failed ({e}), training from scratch")

    hist      = defaultdict(list)
    t0        = time.time()
    stage     = 1
    stage_eps = 0
    best_pdr  = -1.0
    best_ckpt = None

    # Rolling metric deques for curriculum advancement
    roll_collision = deque(maxlen=ROLLING_WINDOW)
    roll_pdr       = deque(maxlen=ROLLING_WINDOW)

    print(f"\n  Stage 1/3: Collision Avoidance")
    print(f"  (advances when rolling collision < {STAGE_ADVANCE[1]['threshold']:.2f} "
          f"over {ROLLING_WINDOW} eps, min {MIN_STAGE_EPS} eps)")

    for ep in range(total_episodes):
        state, _ = env.reset()
        ep_rew   = 0.0
        ep_loss  = []

        while True:
            actions                      = agent.act(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, rewards, nstate, done)
            loss = agent.train_step()
            if loss > 0:
                ep_loss.append(loss)
            state   = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break

        stage_eps += 1
        roll_collision.append(info["avg_collision"])
        roll_pdr.append(info["avg_pdr"])

        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["throughput"].append(info["avg_throughput"])
        hist["fairness"].append(info["fairness"])
        hist["reliability"].append(info["reliability_rate"])
        hist["loss"].append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        hist["stage"].append(stage)

        # ── Curriculum advancement ────────────────────────────────
        if stage < 3:
            metric_hist = roll_collision if stage == 1 else roll_pdr
            advance = should_advance(metric_hist, stage, stage_eps) or (stage_eps >= MAX_STAGE_EPS)

            if advance:
                stage     += 1
                stage_eps  = 0
                roll_collision.clear()
                roll_pdr.clear()
                env.set_curriculum_stage(stage)  # resets reward norm stats
                agent.clear_buffer()              # remove stale transitions
                agent.reset_optimizer(lr=LR * 0.5)  # LR warm restart at half rate
                label = {2: "PDR Optimization", 3: "Full Multi-Objective"}[stage]
                reason = (
                    f"collision={rolling_mean(deque([info['avg_collision']]))*100:.1f}%"
                    if stage == 2 else
                    f"PDR={rolling_mean(deque([info['avg_pdr']])):.3f}"
                )
                print(f"\n  >> Advanced to Stage {stage}/3: {label} at ep {ep} [{reason}]")
                print(f"     Buffer cleared. LR warm-restarted to {LR*0.5:.2e}")

        # ── Logging ───────────────────────────────────────────────
        if ep % 200 == 0:
            elapsed = time.time() - t0
            eps     = agent._get_epsilon()
            buf_sz  = len(agent.per)
            avg_c   = rolling_mean(roll_collision)
            avg_p   = rolling_mean(roll_pdr)
            print(f"    Ep {ep:4d} | Stg {stage} | Rew {ep_rew:7.2f} | "
                  f"PDR {avg_p:.3f} | Coll {avg_c:.3f} | "
                  f"Eps {eps:.3f} | Buf {buf_sz:6d} | t={elapsed:.0f}s")

        # ── Periodic best-model checkpoint ────────────────────────
        if ep > 0 and ep % EVAL_INTERVAL == 0 and stage == 3:
            env_eval = (V2VManhattanEnv if USE_MANHATTAN else V2VEnv)(
                n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
            qmix_res = evaluate_policy(env_eval, lambda e, s: agent.act(s, explore=False), episodes=20)
            cur_pdr  = qmix_res["PDR"]
            if cur_pdr > best_pdr:
                best_pdr  = cur_pdr
                best_ckpt = {
                    "q_network":        copy.deepcopy(agent.q_network.state_dict()),
                    "target_q_network": copy.deepcopy(agent.target_q_network.state_dict()),
                    "mixer":            copy.deepcopy(agent.mixer.state_dict()),
                    "target_mixer":     copy.deepcopy(agent.target_mixer.state_dict()),
                    "n_v2v":            n_v2v,
                    "ep":               ep,
                }
                tag  = "manhattan" if USE_MANHATTAN else "rw"
                path = f"models_sota/qmix_best_{tag}_n{n_v2v}.pth"
                torch.save(best_ckpt, path)
                print(f"    >> Best model (PDR={best_pdr:.3f}) saved → {path}")

    # ── Final evaluation ─────────────────────────────────────────
    # Load best checkpoint so a late-training dip doesn't define reported results
    if best_ckpt is not None:
        agent.q_network.load_state_dict(best_ckpt["q_network"])
        agent.mixer.load_state_dict(best_ckpt["mixer"])
        print(f"  Loaded best QMIX checkpoint (PDR={best_pdr:.3f}, ep={best_ckpt['ep']}) for final eval")
    print("\n  Running final evaluation …")
    env_eval = (V2VManhattanEnv if USE_MANHATTAN else V2VEnv)(
        n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
    qmix_policy  = lambda e, s: agent.act(s, explore=False)
    rand_res     = evaluate_policy(env_eval, random_policy,     episodes=50)
    greedy_res   = evaluate_policy(env_eval, greedy_csi_policy, episodes=50)
    rr_res       = evaluate_policy(env_eval, round_robin_policy, episodes=50)
    qmix_res     = evaluate_policy(env_eval, qmix_policy,       episodes=50)

    # Save final model
    tag        = "manhattan" if USE_MANHATTAN else "rw"
    final_path = f"models_sota/qmix_final_{tag}_n{n_v2v}.pth"
    torch.save({
        "q_network":        agent.q_network.state_dict(),
        "target_q_network": agent.target_q_network.state_dict(),
        "mixer":            agent.mixer.state_dict(),
        "target_mixer":     agent.target_mixer.state_dict(),
        "optimizer":        agent.optimizer.state_dict(),
        "n_v2v":            n_v2v,
        "state_dim":        env.STATE_DIM,
        "n_actions":        env.n_actions,
        "final_evaluation": qmix_res,
    }, final_path)
    print(f"  Final model saved: {final_path}")

    return {
        "agent":       agent,
        "histories":   dict(hist),
        "evaluation":  {
            "Random":     rand_res,
            "Greedy CSI": greedy_res,
            "Round Robin":rr_res,
            "QMIX":       qmix_res,
        },
        "transfer_weights": {
            "q_network": agent.q_network.state_dict(),
            "n_v2v":     n_v2v,
        },
    }


# ─── MAPPO SOTA Training ─────────────────────────────────────────────────────
def train_mappo_sota(
    n_v2v:          int  = 8,
    total_episodes: int  = 4000,
) -> dict:
    """
    MAPPO with adaptive curriculum, value clipping, ortho init, entropy decay.
    """
    scenario = "Manhattan Grid" if USE_MANHATTAN else "Random Walk"
    print(f"\n{'='*60}")
    print(f"  [MAPPO SOTA | {scenario}] n_v2v={n_v2v}  eps={total_episodes}")
    print(f"{'='*60}")

    EnvClass = V2VManhattanEnv if USE_MANHATTAN else V2VEnv
    env = EnvClass(
        n_v2v=n_v2v, n_subchannels=4, episode_len=50,
        curriculum_stage=1,
        normalize_rewards=True,
    )

    agent = MAPPOAgent(
        state_dim      = env.STATE_DIM,
        n_actions      = env.n_actions,
        n_agents       = n_v2v,
        d_model        = 256,
        lr_actor       = 3e-5,    # conservative for stability
        lr_critic      = 1e-4,
        gamma          = 0.99,
        gae_lambda     = 0.97,
        clip_eps       = 0.1,     # tight clipping for stable updates
        entropy_coef   = 0.02,
        entropy_min    = 0.005,   # was 0.001 — too low caused policy collapse in late Stage 3
        epochs         = 4,
        minibatch      = 32,
        grad_clip      = 0.5,
        total_episodes = total_episodes,
        device         = DEVICE,
    )
    print(f"  MAPPO param count: {agent.param_count():,}")

    hist      = defaultdict(list)
    t0        = time.time()
    stage     = 1
    stage_eps = 0
    best_pdr  = -1.0

    roll_collision = deque(maxlen=ROLLING_WINDOW)
    roll_pdr       = deque(maxlen=ROLLING_WINDOW)

    print(f"\n  Stage 1/3: Collision Avoidance")

    for ep in range(total_episodes):
        state, _ = env.reset()
        ep_rew   = 0.0

        while True:
            actions, log_probs, value      = agent.act_with_info(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, log_probs, rewards, value, done)
            state   = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break

        agent.train()

        stage_eps += 1
        roll_collision.append(info["avg_collision"])
        roll_pdr.append(info["avg_pdr"])

        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["throughput"].append(info["avg_throughput"])
        hist["fairness"].append(info["fairness"])
        hist["stage"].append(stage)

        # Adaptive curriculum
        if stage < 3:
            metric_hist = roll_collision if stage == 1 else roll_pdr
            if should_advance(metric_hist, stage, stage_eps) or (stage_eps >= MAX_STAGE_EPS):
                stage     += 1
                stage_eps  = 0
                roll_collision.clear()
                roll_pdr.clear()
                env.set_curriculum_stage(stage)
                label = {2: "PDR Optimization", 3: "Full Multi-Objective"}[stage]
                print(f"\n  >> Advanced to Stage {stage}/3: {label} at ep {ep}")

        if ep % 200 == 0:
            elapsed = time.time() - t0
            ent_c   = agent._current_entropy_coef()
            avg_c   = rolling_mean(roll_collision)
            avg_p   = rolling_mean(roll_pdr)
            print(f"  Ep {ep:4d} | Stg {stage} | Rew {ep_rew:7.2f} | "
                  f"PDR {avg_p:.3f} | Coll {avg_c:.3f} | "
                  f"Ent {ent_c:.4f} | t={elapsed:.0f}s")

        # Best model checkpoint
        if ep > 0 and ep % EVAL_INTERVAL == 0 and stage == 3:
            env_eval = (V2VManhattanEnv if USE_MANHATTAN else V2VEnv)(
                n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
            res = evaluate_policy(env_eval, lambda e, s: agent.act(s), episodes=20)
            if res["PDR"] > best_pdr:
                best_pdr = res["PDR"]
                tag  = "manhattan" if USE_MANHATTAN else "rw"
                path = f"models_sota/mappo_best_{tag}_n{n_v2v}.pth"
                torch.save({
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "n_v2v": n_v2v, "ep": ep,
                }, path)
                print(f"  >> Best MAPPO (PDR={best_pdr:.3f}) saved → {path}")

    # Final evaluation — load best checkpoint so late-stage entropy collapse doesn't hurt results
    tag  = "manhattan" if USE_MANHATTAN else "rw"
    best_path = f"models_sota/mappo_best_{tag}_n{n_v2v}.pth"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        print(f"  Loaded best checkpoint (PDR={best_pdr:.3f}) for final evaluation")
    env_eval = (V2VManhattanEnv if USE_MANHATTAN else V2VEnv)(
        n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
    mappo_res = evaluate_policy(env_eval, lambda e, s: agent.act(s), episodes=50)

    tag  = "manhattan" if USE_MANHATTAN else "rw"
    path = f"models_sota/mappo_final_{tag}_n{n_v2v}.pth"
    torch.save({
        "actor":          agent.actor.state_dict(),
        "critic":         agent.critic.state_dict(),
        "actor_optimizer":agent.actor_opt.state_dict(),
        "critic_optimizer":agent.critic_opt.state_dict(),
        "n_v2v":          n_v2v,
        "state_dim":      env.STATE_DIM,
        "n_actions":      env.n_actions,
        "final_evaluation": mappo_res,
    }, path)
    print(f"  Final MAPPO model saved: {path}")

    return {
        "agent":      agent,
        "histories":  dict(hist),
        "evaluation": {"MAPPO": mappo_res},
    }


# ─── Plot helpers ────────────────────────────────────────────────────────────
def _smooth(x, w=50):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(hist: dict, name: str, n_v2v: int):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{name} Training Curves  (n={n_v2v})", fontsize=14)

    metrics = [
        ("reward",     "Episode Reward",     axes[0, 0]),
        ("pdr",        "Packet Delivery Ratio", axes[0, 1]),
        ("collision",  "Collision Rate",     axes[0, 2]),
        ("throughput", "Throughput (bits/s/Hz)", axes[1, 0]),
        ("fairness",   "Jain's Fairness",    axes[1, 1]),
        ("loss",       "Training Loss",      axes[1, 2]),
    ]
    stage_hist = hist.get("stage", [])

    for key, title, ax in metrics:
        if key not in hist:
            continue
        raw = hist[key]
        ax.plot(raw, alpha=0.2, color="steelblue")
        ax.plot(_smooth(raw), color="steelblue", lw=1.8, label="smoothed")

        # Mark stage transitions
        if stage_hist:
            for ep, stg in enumerate(stage_hist):
                if ep > 0 and stage_hist[ep - 1] != stg:
                    ax.axvline(ep, color="red", lw=1, linestyle="--", alpha=0.7,
                               label=f"→ Stage {stg}")

        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"results/{name.lower().replace(' ', '_')}_n{n_v2v}_sota.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved: {path}")


def print_comparison(results: dict, n_v2v: int):
    print(f"\n{'='*70}")
    print(f"  SOTA RESULTS — n_v2v = {n_v2v}")
    print(f"{'='*70}")
    print(f"  {'Algorithm':<16} {'Reward':>8} {'PDR':>7} {'Collision':>10} "
          f"{'Throughput':>11} {'Fairness':>9}")
    print(f"  {'-'*70}")
    for name, res in results.items():
        r   = res.get("Reward",     0.0)
        pdr = res.get("PDR",        0.0)
        col = res.get("Collision",  0.0)
        thr = res.get("Throughput", 0.0)
        fai = res.get("Fairness",   0.0)
        print(f"  {name:<16} {r:8.2f} {pdr:7.3f} {col:10.3f} {thr:11.3f} {fai:9.3f}")
    print(f"{'='*70}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    DENSITIES      = [8, 16, 24]
    EPISODES       = 5000      # 3000 was too short — both agents stuck in Stage 1/2
    TRAIN_ALL      = True      # n=8 done; now train n=16 and n=24 with transfer learning

    scenario = "Manhattan Grid" if USE_MANHATTAN else "Random Walk"
    print("\n" + "=" * 72)
    print(f"  SOTA TRAINING: QMIX + MAPPO | Scenario: {scenario}")
    print("=" * 72)

    # ── Quick validation: n=8 only ────────────────────────────────────
    print("\n>> Phase 1: n=8 validation run")
    qmix_result_8  = train_qmix_sota(n_v2v=8,  total_episodes=EPISODES)
    mappo_result_8 = train_mappo_sota(n_v2v=8, total_episodes=EPISODES)

    plot_training(qmix_result_8["histories"],  "QMIX",  n_v2v=8)
    plot_training(mappo_result_8["histories"], "MAPPO", n_v2v=8)

    all_results = {**qmix_result_8["evaluation"], **mappo_result_8["evaluation"]}
    print_comparison(all_results, n_v2v=8)

    if not TRAIN_ALL:
        print("\n  Set TRAIN_ALL=True in main() to train n=16 and n=24 with transfer learning.")
        print("  Results saved to ./results/   Models saved to ./models_sota/")
        return

    # ── Full training: n=8 → n=16 → n=24 with transfer ───────────────
    qmix_transfers = {8: qmix_result_8["transfer_weights"]}
    all_qmix  = {8: qmix_result_8}
    all_mappo = {8: mappo_result_8}

    for n in DENSITIES[1:]:
        print(f"\n>> Phase: n={n} (transfer from n={n//2})")
        qmix_r  = train_qmix_sota(n_v2v=n, total_episodes=EPISODES,
                                   transfer_from=qmix_transfers.get(n // 2))
        mappo_r = train_mappo_sota(n_v2v=n, total_episodes=EPISODES)

        all_qmix[n]  = qmix_r
        all_mappo[n] = mappo_r
        qmix_transfers[n] = qmix_r["transfer_weights"]

        plot_training(qmix_r["histories"],  "QMIX",  n_v2v=n)
        plot_training(mappo_r["histories"], "MAPPO", n_v2v=n)
        combined = {**qmix_r["evaluation"], **mappo_r["evaluation"]}
        print_comparison(combined, n_v2v=n)

    # ── Cross-density summary ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CROSS-DENSITY SUMMARY")
    print("=" * 72)
    print(f"  {'Algorithm':<14} {'n=8':>8}  {'n=16':>8}  {'n=24':>8}   (PDR)")
    for alg, results in [("QMIX", all_qmix), ("MAPPO", all_mappo)]:
        row = f"  {alg:<14}"
        for n in DENSITIES:
            ev = results[n]["evaluation"]
            key = "QMIX" if alg == "QMIX" else "MAPPO"
            pdr = ev.get(key, {}).get("PDR", 0.0)
            row += f"  {pdr:8.3f}"
        print(row)
    print("=" * 72)
    print("\n  Models saved to ./models_sota/")


if __name__ == "__main__":
    main()
