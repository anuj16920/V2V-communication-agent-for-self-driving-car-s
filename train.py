"""
Main training script
V2V Resource Allocation — Advanced MARL
==========================================
Trains both Transformer-Dueling-DQN and MAPPO, evaluates against
baselines across densities [8, 16, 24] vehicles.

Usage:
  python train.py

Outputs:
  - Convergence curves (.png)
  - Scalability plots (.png)
  - Baseline comparison bar chart (.png)
  - Printed conference-paper ready table
"""

import os
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from env        import V2VEnv
from agent_dqn  import TransformerDQNAgent
from agent_mappo import MAPPOAgent
from baselines  import random_policy, greedy_csi_policy, round_robin_policy

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────
def evaluate_policy(env, policy_fn, episodes: int = 30) -> dict:
    keys = ["avg_pdr","avg_throughput","avg_collision","avg_sinr",
            "fairness","avg_energy","latency_violation_rate","reliability_rate"]
    acc  = defaultdict(list)
    rew_acc = []

    for _ in range(episodes):
        state, _ = env.reset()
        ep_rew   = 0.0
        step     = 0
        while True:
            actions = policy_fn(env, state)
            state, rewards, done, _, info = env.step(actions)
            ep_rew += float(np.mean(rewards))
            step   += 1
            if done:
                break
        rew_acc.append(ep_rew)
        for k in keys:
            acc[k].append(info[k])

    result = {"Reward": float(np.mean(rew_acc))}
    metric_map = {
        "avg_pdr":               "PDR",
        "avg_throughput":        "Throughput",
        "avg_collision":         "Collision",
        "avg_sinr":              "SINR",
        "fairness":              "Fairness",
        "avg_energy":            "Energy",
        "latency_violation_rate":"LatencyViolation",
        "reliability_rate":      "Reliability",
    }
    for k, label in metric_map.items():
        result[label] = float(np.mean(acc[k]))
    return result


# ─────────────────────────────────────────────
# DQN Training
# ─────────────────────────────────────────────
def train_dqn(n_v2v: int, episodes: int = 1000) -> dict:
    print(f"\n{'='*60}")
    print(f"  [DQN] Training  n_v2v={n_v2v}  episodes={episodes}")
    print(f"{'='*60}")

    env = V2VEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50)

    agent = TransformerDQNAgent(
        state_dim    = env.STATE_DIM,
        n_actions    = env.n_actions,
        n_agents     = n_v2v,
        d_model      = 128,
        n_heads      = 4,
        n_layers     = 2,
        d_ff         = 256,
        dropout      = 0.1,
        lr           = 1e-4,         # FIXED
        gamma        = 0.99,
        batch_size   = 512,          # FIXED
        n_steps      = 3,            # FIXED
        tau          = 0.005,
        grad_clip    = 10.0,
        update_freq  = 2,            # FIXED: train more frequently
        per_capacity = 200_000,
        epsilon_start  = 1.0,        # FIXED
        epsilon_end    = 0.01,       # FIXED: lower final epsilon
        epsilon_steps  = 80_000,     # FIXED: faster decay
        device       = DEVICE,
    )

    print(f"  DQN param count: {agent.param_count():,}")

    hist = defaultdict(list)
    t0   = time.time()

    for ep in range(episodes):
        state, _ = env.reset()
        ep_rew   = 0.0
        ep_loss  = []

        while True:
            actions = agent.act(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, rewards, nstate, done)
            loss = agent.train_step()
            if loss > 0:
                ep_loss.append(loss)
            state   = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break

        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["throughput"].append(info["avg_throughput"])
        hist["collision"].append(info["avg_collision"])
        hist["sinr"].append(info["avg_sinr"])
        hist["fairness"].append(info["fairness"])
        hist["energy"].append(info["avg_energy"])
        hist["latency"].append(info["latency_violation_rate"])
        hist["reliability"].append(info["reliability_rate"])
        hist["loss"].append(float(np.mean(ep_loss)) if ep_loss else 0.0)

        if ep % 100 == 0:
            elapsed = time.time() - t0
            eps = agent._get_epsilon()
            print(
                f"  Ep {ep:4d} | Rew {ep_rew:8.2f} | "
                f"PDR {info['avg_pdr']:.3f} | Thr {info['avg_throughput']:.3f} | "
                f"Coll {info['avg_collision']:.3f} | Fair {info['fairness']:.3f} | "
                f"Eps {eps:.3f} | Loss {hist['loss'][-1]:.4f} | t={elapsed:.0f}s"
            )

    # ── Evaluation (greedy, no noise) ─────────────────────────────
    def dqn_policy(eval_env, state):
        actions = np.zeros(eval_env.n_v2v, dtype=np.int64)
        agent.online_net.eval()
        for i in range(eval_env.n_v2v):
            s = torch.FloatTensor(state[i:i+1]).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.online_net(s)
            actions[i] = q.squeeze().argmax().item()
        return actions

    rand_res   = evaluate_policy(env, random_policy,      episodes=30)
    greedy_res = evaluate_policy(env, greedy_csi_policy,  episodes=30)
    rr_res     = evaluate_policy(env, round_robin_policy,  episodes=30)
    dqn_res    = evaluate_policy(env, dqn_policy,          episodes=30)

    # Save model
    model_path = f"models/dqn_n{n_v2v}.pth"
    torch.save({
        'online_net': agent.online_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'n_v2v': n_v2v,
        'state_dim': env.STATE_DIM,
        'n_actions': env.n_actions,
        'final_evaluation': dqn_res,
    }, model_path)
    print(f"  Model saved: {model_path}")

    return {
        "agent":      agent,
        "histories":  dict(hist),
        "evaluation": {
            "Random":        rand_res,
            "Greedy CSI":    greedy_res,
            "Round Robin":   rr_res,
            "Transformer-DQN": dqn_res,
        },
    }


# ─────────────────────────────────────────────
# MAPPO Training
# ─────────────────────────────────────────────
def train_mappo(n_v2v: int, episodes: int = 1000) -> dict:
    print(f"\n{'='*60}")
    print(f"  [MAPPO] Training  n_v2v={n_v2v}  episodes={episodes}")
    print(f"{'='*60}")

    env = V2VEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50)

    agent = MAPPOAgent(
        state_dim    = env.STATE_DIM,
        n_actions    = env.n_actions,
        n_agents     = n_v2v,
        d_model      = 128,
        lr_actor     = 3e-4,
        lr_critic    = 1e-3,
        gamma        = 0.99,
        gae_lambda   = 0.95,
        clip_eps     = 0.2,
        entropy_coef = 0.02,    # FIXED: increased from 0.01
        epochs       = 10,
        minibatch    = 64,
        device       = DEVICE,
    )

    print(f"  MAPPO param count: {agent.param_count():,}")

    hist = defaultdict(list)
    t0   = time.time()

    for ep in range(episodes):
        state, _ = env.reset()
        ep_rew   = 0.0

        while True:
            actions, log_probs, value = agent.act_with_info(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, log_probs, rewards, value, done)
            state   = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break

        train_logs = agent.train()

        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["throughput"].append(info["avg_throughput"])
        hist["collision"].append(info["avg_collision"])
        hist["sinr"].append(info["avg_sinr"])
        hist["fairness"].append(info["fairness"])
        hist["energy"].append(info["avg_energy"])
        hist["latency"].append(info["latency_violation_rate"])
        hist["reliability"].append(info["reliability_rate"])
        hist["actor_loss"].append(train_logs.get("actor_loss", 0.0))

        if ep % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  Ep {ep:4d} | Rew {ep_rew:8.2f} | "
                f"PDR {info['avg_pdr']:.3f} | Thr {info['avg_throughput']:.3f} | "
                f"Fair {info['fairness']:.3f} | t={elapsed:.0f}s"
            )

    def mappo_policy(eval_env, state):
        return agent.act(state)

    mappo_res = evaluate_policy(env, mappo_policy, episodes=30)

    # Save model
    model_path = f"models/mappo_n{n_v2v}.pth"
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'actor_optimizer': agent.actor_opt.state_dict(),
        'critic_optimizer': agent.critic_opt.state_dict(),
        'n_v2v': n_v2v,
        'state_dim': env.STATE_DIM,
        'n_actions': env.n_actions,
        'final_evaluation': mappo_res,
    }, model_path)
    print(f"  Model saved: {model_path}")

    return {
        "agent":      agent,
        "histories":  dict(hist),
        "evaluation": {"MAPPO": mappo_res},
    }


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
def smooth(data, window: int = 20) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def plot_convergence(dqn_hist: dict, mappo_hist: dict, n_v2v: int):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Training Convergence  (n_v2v={n_v2v})", fontsize=14, fontweight='bold')

    keys = [
        ("reward",      "Reward",            "Episode Reward"),
        ("pdr",         "PDR",               "Packet Delivery Ratio"),
        ("throughput",  "Throughput",        "Rate (bps/Hz)"),
        ("collision",   "Collision Rate",    "Collision Rate"),
        ("sinr",        "Avg SINR",          "SINR"),
        ("fairness",    "Jain's Fairness",   "Fairness Index"),
        ("energy",      "Energy Usage",      "Energy"),
        ("latency",     "Latency Violation", "Violation Rate"),
        ("reliability", "Reliability",       "Rate SINR ≥ 1"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, keys):
        if key in dqn_hist:
            ax.plot(smooth(dqn_hist[key]), label="Transformer-DQN", color="#2196F3", lw=1.5)
        if key in mappo_hist:
            ax.plot(smooth(mappo_hist[key]), label="MAPPO", color="#FF5722", lw=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"results/convergence_n{n_v2v}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_scalability(all_dqn: dict, all_mappo: dict, densities: list):
    metrics = ["PDR", "Throughput", "Fairness", "Reliability", "Collision"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scalability: Performance vs Vehicle Density", fontsize=14, fontweight='bold')

    methods_colors = {
        "Random":           ("#9E9E9E", "o"),
        "Greedy CSI":       ("#FFC107", "s"),
        "Round Robin":      ("#4CAF50", "^"),
        "Transformer-DQN":  ("#2196F3", "D"),
        "MAPPO":            ("#FF5722", "P"),
    }

    for ax, metric in zip(axes.flat, metrics):
        for method, (color, marker) in methods_colors.items():
            vals = []
            for d in densities:
                if method in ["Random", "Greedy CSI", "Round Robin", "Transformer-DQN"]:
                    src = all_dqn.get(d, {}).get("evaluation", {})
                else:
                    src = all_mappo.get(d, {}).get("evaluation", {})
                if method in src:
                    vals.append(src[method].get(metric, np.nan))
                else:
                    vals.append(np.nan)
            if any(not np.isnan(v) for v in vals):
                ax.plot(densities, vals, marker=marker, color=color, label=method, lw=2, markersize=8)

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("# Vehicles")
        ax.set_ylabel(metric)
        ax.set_xticks(densities)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    path = "results/scalability.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_bar_comparison(all_results_n8: dict):
    metrics  = ["PDR", "Throughput", "Fairness", "Reliability", "Collision", "LatencyViolation"]
    methods  = list(all_results_n8.keys())
    colors   = ["#9E9E9E","#FFC107","#4CAF50","#2196F3","#FF5722","#9C27B0"][:len(methods)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Baseline Comparison (n_v2v = 8)", fontsize=13, fontweight='bold')

    for ax, metric in zip(axes.flat, metrics):
        vals = [all_results_n8[m].get(metric, 0.0) for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=10)
        ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = "results/bar_comparison_n8.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────
def main():
    DENSITIES = [8, 16, 24]
    EPISODES  = 1000  # FIXED: increased from 600

    all_dqn   = {}
    all_mappo = {}

    for n in DENSITIES:
        all_dqn[n]   = train_dqn(n_v2v=n, episodes=EPISODES)
        all_mappo[n] = train_mappo(n_v2v=n, episodes=EPISODES)

        plot_convergence(
            all_dqn[n]["histories"],
            all_mappo[n]["histories"],
            n_v2v=n,
        )

    plot_scalability(all_dqn, all_mappo, DENSITIES)

    # Combined results for n=8 bar chart
    combined_n8 = {}
    combined_n8.update(all_dqn[8]["evaluation"])
    combined_n8.update(all_mappo[8]["evaluation"])
    plot_bar_comparison(combined_n8)

    # ── Conference-paper table ────────────────────────────────────────
    print("\n" + "="*100)
    print("  FINAL CONFERENCE-PAPER RESULTS")
    print("="*100)

    metrics_print = ["Reward","PDR","Throughput","Collision","SINR","Fairness","Energy","LatencyViolation","Reliability"]

    for n in DENSITIES:
        print(f"\n  ── Vehicle Density n={n} ──")
        eval_dqn   = all_dqn[n]["evaluation"]
        eval_mappo = all_mappo[n]["evaluation"]
        all_eval   = {**eval_dqn, **eval_mappo}

        header = f"{'Method':<20}" + "".join(f"{m:>17}" for m in metrics_print)
        print(f"  {header}")
        print("  " + "-"*len(header))

        for method, vals in all_eval.items():
            row = f"  {method:<20}" + "".join(f"{vals.get(m, 0.0):>17.4f}" for m in metrics_print)
            print(row)

    print("\n" + "="*100)
    print("  KEY HIGHLIGHTS (n=8, Transformer-DQN vs baselines)")
    print("="*100)
    best = all_dqn[8]["evaluation"].get("Transformer-DQN", {})
    rand = all_dqn[8]["evaluation"].get("Random", {})
    for m in ["PDR","Throughput","Fairness","Reliability","Collision","LatencyViolation"]:
        bv = best.get(m, 0.0)
        rv = rand.get(m, 0.0)
        delta = ((bv - rv) / max(abs(rv), 1e-6)) * 100
        sign  = "+" if delta > 0 else ""
        print(f"  {m:<22}: DQN={bv:.4f}  Random={rv:.4f}  ({sign}{delta:.1f}% vs Random)")

    print("\n  [Done] All results saved to ./results/")


if __name__ == "__main__":
    main()
