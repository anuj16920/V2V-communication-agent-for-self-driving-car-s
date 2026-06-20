# -*- coding: utf-8 -*-
"""
PGAT-MAPPO Training Script  (Mixed-Density)
============================================
Trains the Predictive Graph Attention MAPPO on Manhattan Grid V2V across a
SWEEP of vehicle densities {20, 40, 60, 80, 100, 120} with a SINGLE model.

Why a single mixed-density model:
  PGAT-MAPPO is N-agnostic by construction (graph attention + mean-pool critic),
  so one policy can be trained on, and deployed at, ANY vehicle count. Each
  episode samples a density from the target set; the same network handles all of
  them. This is both more efficient than six per-density specialists (~9h vs
  ~47h) and a stronger result: "one policy, 20-120 vehicles, no retraining".

Curriculum:
  Fixed-fraction 3-stage curriculum (Stage 1: collision avoidance, Stage 2: PDR,
  Stage 3: full objective). Fixed fractions are used instead of metric-based
  advancement because the advancement metric (collision/PDR) is non-stationary
  across mixed densities (n=120 has structurally higher collision than n=20),
  which makes a single threshold unreliable.

Usage:
    python train_pgat.py                 # mixed-density training (default)
    # edit MODE below to 'specialist' to train one model per density instead
"""

import os, time, warnings
import numpy as np
import torch
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
warnings.filterwarnings("ignore")

from env_manhattan    import V2VManhattanEnv
from agent_graph_mappo import PGATMAPPOAgent
from networks         import MAPPOActor
from baselines        import random_policy, greedy_csi_policy, round_robin_policy

os.makedirs("results",     exist_ok=True)
os.makedirs("models_pgat", exist_ok=True)

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Target density sweep ─────────────────────────────────────────────────────
TARGET_DENSITIES = [20, 40, 60, 80, 100, 120]

# ─── Curriculum (fixed-fraction, robust to mixed-density metric noise) ─────────
STAGE_FRACTIONS = [0.30, 0.30, 0.40]   # Stage 1, 2, 3 share of total episodes
EVAL_INTERVAL   = 500                  # episodes between best-checkpoint evals
EVAL_DENSITIES  = [20, 60, 120]        # subset used for periodic best-model eval


def rolling_mean(d: deque) -> float:
    return float(np.mean(list(d))) if d else 0.0


def _stage_for_episode(ep: int, total: int) -> int:
    """Return curriculum stage (1..3) for episode `ep` under fixed fractions."""
    f1 = STAGE_FRACTIONS[0]
    f2 = STAGE_FRACTIONS[0] + STAGE_FRACTIONS[1]
    frac = ep / max(1, total)
    if frac < f1:
        return 1
    if frac < f2:
        return 2
    return 3


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_pgat(env, agent, n_eps=30) -> dict:
    """Evaluate a PGAT agent. Uses agent.eval_temperature (set via set_eval_temperature)."""
    acc = defaultdict(list)
    for _ in range(n_eps):
        obs, info = env.reset()
        done = False
        while not done:
            actions = agent.act_eval(obs, info)
            obs, _, done, _, info = env.step(actions)
        for k, v in info.items():
            if isinstance(v, (int, float)):
                acc[k].append(v)
    return {
        "PDR":         float(np.mean(acc["avg_pdr"])),
        "Collision":   float(np.mean(acc["avg_collision"])),
        "Throughput":  float(np.mean(acc["avg_throughput"])),
        "Fairness":    float(np.mean(acc["fairness"])),
        "Reliability": float(np.mean(acc["reliability_rate"])),
    }


def find_best_temperature(agent, n_v2v=40, n_eps=20) -> float:
    """
    Grid-search eval temperature on one mid-range density.
    Pure argmax (T=1.0) causes channel collapse when agents have correlated
    observations (same street -> same CSI -> same argmax).
    Returns the temperature that maximises PDR.
    """
    env = V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4,
                          episode_len=50, curriculum_stage=3)
    best_t, best_pdr = 1.0, -1.0
    print(f"\n  Temperature search at n={n_v2v}:")
    for t in [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]:
        agent.set_eval_temperature(t)
        res = evaluate_pgat(env, agent, n_eps=n_eps)
        mark = " <-- best" if res["PDR"] > best_pdr else ""
        print(f"    T={t:.1f}  PDR={res['PDR']:.3f}  Coll={res['Collision']:.3f}{mark}")
        if res["PDR"] > best_pdr:
            best_pdr = res["PDR"]
            best_t   = t
    print(f"  Best temperature: {best_t}  (PDR={best_pdr:.3f})")
    agent.set_eval_temperature(best_t)
    return best_t


def evaluate_all_densities(agent, densities, n_eps=30) -> dict:
    """Evaluate the agent at each density; return {n: metrics}."""
    out = {}
    for n in densities:
        env = V2VManhattanEnv(n_v2v=n, n_subchannels=4,
                              episode_len=50, curriculum_stage=3)
        out[n] = evaluate_pgat(env, agent, n_eps=n_eps)
    return out


# ─── Mixed-density training ───────────────────────────────────────────────────

def train_pgat_mixed(densities=TARGET_DENSITIES, total_episodes=6000):
    print(f"\n{'='*64}")
    print(f"  PGAT-MAPPO | Manhattan Grid | MIXED-DENSITY {densities}")
    print(f"  total episodes={total_episodes}  (~{total_episodes//len(densities)}/density)")
    print(f"{'='*64}")

    # Env pool: one per density (preserves per-density reward-norm state)
    envs = {n: V2VManhattanEnv(n_v2v=n, n_subchannels=4, episode_len=50,
                               curriculum_stage=1, normalize_rewards=True)
            for n in densities}

    # Agent constructed once (N-agnostic across all densities)
    agent = PGATMAPPOAgent(
        state_dim      = 32,
        n_actions      = 16,
        n_agents       = max(densities),   # vestigial; networks are N-agnostic
        d_model        = 128,
        n_heads        = 4,
        n_layers       = 3,
        lr_actor       = 3e-5,
        lr_critic      = 1e-4,
        clip_eps       = 0.1,
        entropy_coef   = 0.02,
        entropy_min    = 0.005,
        epochs         = 4,
        minibatch      = 16,
        grad_clip      = 0.5,
        total_episodes = total_episodes,
        device         = DEVICE,
    )
    print(f"  PGAT-MAPPO param count: {agent.param_count():,}")

    rng       = np.random.default_rng(SEED)
    hist      = defaultdict(list)
    t0        = time.time()
    stage     = 1
    best_pdr  = -1.0
    # Per-density rolling logs for readable progress
    roll_pdr  = {n: deque(maxlen=50) for n in densities}
    roll_coll = {n: deque(maxlen=50) for n in densities}

    stage_labels = {1: "Collision Avoidance", 2: "PDR Optimization",
                    3: "Full Multi-Objective"}
    print(f"\n  Stage 1/3: {stage_labels[1]}  (fixed-fraction curriculum)")

    for ep in range(total_episodes):
        # ── Curriculum stage (fixed fraction); sync ALL envs on change ──
        new_stage = _stage_for_episode(ep, total_episodes)
        if new_stage != stage:
            stage = new_stage
            for e in envs.values():
                e.set_curriculum_stage(stage)
            print(f"\n  >> Stage {stage}/3: {stage_labels[stage]} at ep {ep} "
                  f"(t={time.time()-t0:.0f}s)")

        # ── Sample density for this episode ─────────────────────────────
        n   = int(rng.choice(densities))
        env = envs[n]

        obs, info = env.reset()
        ep_rew    = 0.0
        done      = False
        while not done:
            actions, log_probs, value = agent.act_with_info(obs, info)
            next_obs, rewards, done, _, next_info = env.step(actions)
            agent.store(obs, actions, log_probs, rewards, value, done, info)
            obs, info = next_obs, next_info
            ep_rew   += float(np.mean(rewards))

        logs = agent.train()

        roll_pdr[n].append(info["avg_pdr"])
        roll_coll[n].append(info["avg_collision"])
        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["throughput"].append(info["avg_throughput"])
        hist["fairness"].append(info["fairness"])
        hist["loss"].append(logs.get("actor_loss", 0.0))
        hist["density"].append(n)

        # ── Periodic best-model eval (mean PDR over subset) ─────────────
        if ep > 0 and ep % EVAL_INTERVAL == 0 and stage == 3:
            sub = evaluate_all_densities(agent, EVAL_DENSITIES, n_eps=10)
            mean_pdr = float(np.mean([sub[k]["PDR"] for k in EVAL_DENSITIES]))
            if mean_pdr > best_pdr:
                best_pdr = mean_pdr
                torch.save({"actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "densities": densities, "ep": ep,
                            "mean_pdr": best_pdr},
                           "models_pgat/pgat_mixed_best.pth")
                tag = "  ".join(f"n{k}={sub[k]['PDR']:.3f}" for k in EVAL_DENSITIES)
                print(f"  >> Best PGAT (mean PDR={best_pdr:.3f} | {tag}) saved")

        # ── Progress log ────────────────────────────────────────────────
        if ep % 200 == 0:
            pdr_str = " ".join(f"n{k}:{rolling_mean(roll_pdr[k]):.2f}"
                               for k in densities)
            print(f"  Ep {ep:4d} | Stg {stage} | n={n:3d} | "
                  f"Rew {ep_rew:7.2f} | Ent {agent._entropy_coef():.4f} | "
                  f"t={time.time()-t0:.0f}s")
            print(f"           PDR[{pdr_str}]")

    # ── Final: load best checkpoint, evaluate at ALL densities ──────────
    if os.path.exists("models_pgat/pgat_mixed_best.pth"):
        ckpt = torch.load("models_pgat/pgat_mixed_best.pth", map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        print(f"\n  Loaded best checkpoint (mean PDR={ckpt.get('mean_pdr', -1):.3f}) for final eval")

    # Find best eval temperature (argmax collapses all agents to same channel
    # when graph embeddings are correlated at episode initialization)
    find_best_temperature(agent, n_v2v=40, n_eps=20)

    final = evaluate_all_densities(agent, densities, n_eps=50)
    torch.save({"actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "densities": densities, "final_eval": final},
               "models_pgat/pgat_mixed_final.pth")
    print(f"\n  Final PGAT model saved: models_pgat/pgat_mixed_final.pth")

    return {"agent": agent, "histories": dict(hist), "evaluation": final}


# ─── Single-density specialist (optional alternative) ─────────────────────────

def train_pgat(n_v2v=20, total_episodes=5000):
    """Train one specialist PGAT model at a single density (fixed-fraction curriculum)."""
    print(f"\n{'='*60}")
    print(f"  PGAT-MAPPO | Manhattan Grid | SPECIALIST n_v2v={n_v2v}  eps={total_episodes}")
    print(f"{'='*60}")

    env = V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50,
                          curriculum_stage=1, normalize_rewards=True)
    agent = PGATMAPPOAgent(state_dim=32, n_actions=16, n_agents=n_v2v,
                           d_model=128, n_heads=4, n_layers=3,
                           lr_actor=3e-5, lr_critic=1e-4, clip_eps=0.1,
                           entropy_coef=0.02, entropy_min=0.005, epochs=4,
                           minibatch=16, grad_clip=0.5,
                           total_episodes=total_episodes, device=DEVICE)
    print(f"  PGAT-MAPPO param count: {agent.param_count():,}")

    hist = defaultdict(list); t0 = time.time(); stage = 1; best_pdr = -1.0
    stage_labels = {1: "Collision Avoidance", 2: "PDR Optimization", 3: "Full Multi-Objective"}
    print(f"\n  Stage 1/3: {stage_labels[1]}")

    for ep in range(total_episodes):
        new_stage = _stage_for_episode(ep, total_episodes)
        if new_stage != stage:
            stage = new_stage
            env.set_curriculum_stage(stage)
            print(f"\n  >> Stage {stage}/3: {stage_labels[stage]} at ep {ep}")

        obs, info = env.reset(); ep_rew = 0.0; done = False
        while not done:
            actions, log_probs, value = agent.act_with_info(obs, info)
            next_obs, rewards, done, _, next_info = env.step(actions)
            agent.store(obs, actions, log_probs, rewards, value, done, info)
            obs, info = next_obs, next_info
            ep_rew += float(np.mean(rewards))
        logs = agent.train()

        hist["reward"].append(ep_rew); hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["throughput"].append(info["avg_throughput"])
        hist["fairness"].append(info["fairness"]); hist["loss"].append(logs.get("actor_loss", 0.0))

        if ep > 0 and ep % EVAL_INTERVAL == 0 and stage == 3:
            res = evaluate_pgat(V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4,
                                                episode_len=50, curriculum_stage=3),
                                agent, n_eps=20)
            if res["PDR"] > best_pdr:
                best_pdr = res["PDR"]
                torch.save({"actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "n_v2v": n_v2v, "ep": ep},
                           f"models_pgat/pgat_best_n{n_v2v}.pth")
                print(f"  >> Best PGAT (PDR={best_pdr:.3f}) saved")

        if ep % 200 == 0:
            print(f"  Ep {ep:4d} | Stg {stage} | Rew {ep_rew:7.2f} | "
                  f"PDR {info['avg_pdr']:.3f} | Coll {info['avg_collision']:.3f} | "
                  f"Ent {agent._entropy_coef():.4f} | t={time.time()-t0:.0f}s")

    best_path = f"models_pgat/pgat_best_n{n_v2v}.pth"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
    final = evaluate_pgat(V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4,
                                          episode_len=50, curriculum_stage=3),
                          agent, n_eps=50)
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
                "n_v2v": n_v2v, "eval": final}, f"models_pgat/pgat_final_n{n_v2v}.pth")
    print(f"\n  Final PGAT saved: models_pgat/pgat_final_n{n_v2v}.pth")
    return {"agent": agent, "histories": dict(hist), "evaluation": final}


# ─── Comparison table across densities ────────────────────────────────────────

def print_density_comparison(pgat_eval, densities):
    """
    Compare PGAT (trained) vs MAPPO (zero-shot from n=8) vs Greedy-CSI / Random
    across all densities. QMIX omitted: its mixer is N-specific (only n=8/16/24).
    """
    # Load N-agnostic MAPPO actor (trained at n=8, generalizes to any N)
    mappo_actor = None
    mp_path = "models_sota/mappo_best_manhattan_n8.pth"
    if os.path.exists(mp_path):
        ck = torch.load(mp_path, map_location=DEVICE)
        mappo_actor = MAPPOActor(state_dim=32, n_actions=16, d_model=256).to(DEVICE)
        mappo_actor.load_state_dict(ck["actor"]); mappo_actor.eval()

    def run_policy(env, fn, n_eps=40):
        acc = defaultdict(list)
        for _ in range(n_eps):
            obs, _ = env.reset(); done = False
            while not done:
                obs, _, done, _, info = env.step(fn(env, obs))
            acc["pdr"].append(info["avg_pdr"]); acc["coll"].append(info["avg_collision"])
        return float(np.mean(acc["pdr"])), float(np.mean(acc["coll"]))

    @torch.no_grad()
    def mappo_fn(env, obs):
        s = torch.FloatTensor(obs).to(DEVICE)
        return mappo_actor.get_dist(s).logits.argmax(-1).cpu().numpy()

    print(f"\n{'='*72}")
    print(f"  PGAT-MAPPO DENSITY SWEEP RESULTS  (PDR | Collision)")
    print(f"{'='*72}")
    print(f"  {'n_v2v':<8}{'Random':>16}{'Greedy-CSI':>16}{'MAPPO(zs)':>16}{'PGAT(ours)':>16}")
    print(f"  {'-'*70}")
    for n in densities:
        env = V2VManhattanEnv(n_v2v=n, n_subchannels=4, episode_len=50, curriculum_stage=3)
        rp, rc = run_policy(env, random_policy)
        gp, gc = run_policy(env, greedy_csi_policy)
        if mappo_actor is not None:
            mp, mc = run_policy(env, mappo_fn)
            mstr = f"{mp:.3f}/{mc:.3f}"
        else:
            mstr = "  --  "
        pe = pgat_eval[n]
        rstr = f"{rp:.3f}/{rc:.3f}"
        gstr = f"{gp:.3f}/{gc:.3f}"
        pstr = f"{pe['PDR']:.3f}/{pe['Collision']:.3f}"
        print(f"  {n:<8}{rstr:>16}{gstr:>16}{mstr:>16}{pstr:>16}")
    print(f"{'='*72}")


# ─── Main ──────────────────────────────────────────────────────────────────────

MODE = "mixed"   # "mixed" = one model over all densities; "specialist" = one per density

if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    if MODE == "mixed":
        result = train_pgat_mixed(TARGET_DENSITIES, total_episodes=6000)
        print_density_comparison(result["evaluation"], TARGET_DENSITIES)
    else:
        all_eval = {}
        for n in TARGET_DENSITIES:
            r = train_pgat(n_v2v=n, total_episodes=5000)
            all_eval[n] = r["evaluation"]
        print_density_comparison(all_eval, TARGET_DENSITIES)
