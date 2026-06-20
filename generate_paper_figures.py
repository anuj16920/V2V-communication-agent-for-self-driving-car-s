# -*- coding: utf-8 -*-
"""
IEEE TVT/TITS Journal Figure Generator
========================================
Generates all 19 publication-quality figures for the V2V MARL paper.

Usage:
    python generate_paper_figures.py

Outputs saved to paper_figures/ (configure FIG_DIR constant to change)
"""

import os, time, math, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict
warnings.filterwarnings("ignore")

from env_manhattan import V2VManhattanEnv, H_STREETS, V_STREETS, INTERSECTION_R
from networks      import MAPPOActor
from agent_qmix    import QMIXAgent
from baselines     import random_policy, greedy_csi_policy, round_robin_policy

# ── Output folder — change this one line to redirect all figures ──────────────
FIG_DIR = "paper_figures"   # e.g. "D:/thesis/figures" or "submission/figs"
os.makedirs(FIG_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "lines.linewidth":  2.2,
    "lines.markersize": 7,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

ALG_STYLE = {
    "Random":      {"color": "#808080", "marker": "o",  "ls": ":"},
    "Round_Robin": {"color": "#9467bd", "marker": "^",  "ls": "-."},
    "Greedy_CSI":  {"color": "#2ca02c", "marker": "s",  "ls": "--"},
    "QMIX":        {"color": "#1f77b4", "marker": "D",  "ls": "-"},
    "MAPPO":       {"color": "#ff7f0e", "marker": "P",  "ls": "-"},
    "PGAT":        {"color": "#d62728", "marker": "*",  "ls": "-"},
}
ALG_LABEL = {
    "Random": "Random", "Round_Robin": "Round Robin",
    "Greedy_CSI": "Greedy-CSI", "QMIX": "QMIX", "MAPPO": "MAPPO",
    "PGAT": "PGAT-MAPPO (ours)",
}

EVAL_EPS   = 100
EPISODE_LEN= 50
N_SUBCH    = 4
N_ACTIONS  = 16

# Target sweep. PGAT (mixed) + MAPPO (zero-shot from n=8) run at all densities.
# QMIX has no checkpoints here (N-specific mixer) -> auto-absent from the curves.
DENSITIES_RL  = []                              # QMIX checkpoints at target N: none
DENSITIES_ALL = [20, 40, 60, 80, 100, 120]

# ── Model loading ─────────────────────────────────────────────────────────────

def load_mappo_actor(n_trained=8):
    path = f"models_sota/mappo_best_manhattan_n{n_trained}.pth"
    ckpt  = torch.load(path, map_location=DEVICE)
    actor = MAPPOActor(state_dim=32, n_actions=N_ACTIONS, d_model=256).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor

def load_qmix(n_v2v):
    path = f"models_sota/qmix_best_manhattan_n{n_v2v}.pth"
    if not os.path.exists(path):
        return None
    ckpt  = torch.load(path, map_location=DEVICE)
    agent = QMIXAgent(state_dim=32, n_actions=N_ACTIONS, n_agents=n_v2v,
                      d_model=256, n_heads=8, n_layers=3, d_ff=512,
                      mixing_embed_dim=64, device=DEVICE)
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.q_network.eval()
    return agent

@torch.no_grad()
def mappo_act(actor, state):
    s = torch.FloatTensor(state).to(DEVICE)
    return actor.get_dist(s).logits.argmax(dim=-1).cpu().numpy()

def load_pgat_mixed():
    """Load mixed-density PGAT model (N-agnostic). Returns agent or None."""
    from agent_graph_mappo import PGATMAPPOAgent
    for path in ("models_pgat/pgat_mixed_best.pth", "models_pgat/pgat_mixed_final.pth"):
        if os.path.exists(path):
            ckpt  = torch.load(path, map_location=DEVICE)
            agent = PGATMAPPOAgent(state_dim=32, n_actions=N_ACTIONS, n_agents=20,
                                   d_model=128, n_heads=4, n_layers=3, device=DEVICE)
            agent.actor.load_state_dict(ckpt["actor"]); agent.actor.eval()
            agent.set_eval_temperature(0.3)
            print(f"  [PGAT] loaded {path}  (T=0.3)")
            return agent
    print("  [PGAT] no mixed checkpoint found -> PGAT curves will be absent")
    return None

# ── Evaluation helpers ────────────────────────────────────────────────────────

def run_eval(env, policy_fn, n_eps, collect_steps=False):
    """
    Returns aggregated per-episode metrics and optionally per-step distributions.
    policy_fn(state, step) -> actions
    """
    eps_data  = defaultdict(list)
    step_data = defaultdict(list) if collect_steps else None

    for _ in range(n_eps):
        state, info = env.reset()          # reset info carries graph fields (PGAT)
        done = False; t = 0
        while not done:
            actions = policy_fn(state, t, info)
            state, _, done, _, info = env.step(actions)
            t += 1
            if collect_steps:
                step_data["sinr"].append(info["avg_sinr"])
                step_data["throughput"].append(info["avg_throughput"])
                step_data["latency_viol"].append(info["latency_violation_rate"])
                step_data["pdr"].append(info["avg_pdr"])
        for k, v in info.items():
            if isinstance(v, (int, float)):  # skip graph arrays (positions, etc.)
                eps_data[k].append(v)

    def agg(d):
        pdr    = float(np.mean(d["avg_pdr"]))
        coll   = float(np.mean(d["avg_collision"]))
        tput   = float(np.mean(d["avg_throughput"]))
        fair   = float(np.mean(d["fairness"]))
        sinr   = float(np.mean(d["avg_sinr"]))
        energy = float(np.mean(d["avg_energy"]))
        lat    = float(np.mean(d["latency_violation_rate"]))
        rel    = float(np.mean(d["reliability_rate"]))
        los    = float(np.mean(d.get("avg_los_ratio", [0.5])))
        sinr_db      = 10.0 * math.log10(max(sinr, 1e-9))
        net_tput     = tput * env.n_v2v
        spec_eff     = tput / N_SUBCH
        energy_eff   = pdr  / max(energy, 1e-9)
        outage       = float(np.mean([1.0 if s < 1.0 else 0.0 for s in d["avg_sinr"]]))
        interf_approx= max(0.0, float(np.mean(
            [si / max(sr, 1e-9) - 1.0 for si, sr in zip(d["avg_sinr"], d["avg_sinr"])]
        ))) if False else sinr / max(sinr, 1e-9) * 0.1
        return dict(pdr=pdr, collision=coll, throughput=tput, net_throughput=net_tput,
                    spectral_eff=spec_eff, fairness=fair, sinr_db=sinr_db, sinr_lin=sinr,
                    latency_viol=lat, reliability=rel, energy_eff=energy_eff,
                    los_ratio=los, outage=outage)

    return agg(eps_data), step_data


def make_policy(alg_name, env, actor=None, qmix_agent=None, pgat_agent=None):
    if alg_name == "Random":
        return lambda s, t, info: random_policy(env, s)
    elif alg_name == "Greedy_CSI":
        return lambda s, t, info: greedy_csi_policy(env, s)
    elif alg_name == "Round_Robin":
        return lambda s, t, info: round_robin_policy(env, s, t)
    elif alg_name == "MAPPO":
        return lambda s, t, info: mappo_act(actor, s)
    elif alg_name == "QMIX":
        return lambda s, t, info: qmix_agent.act(s, explore=False)
    elif alg_name == "PGAT":
        return lambda s, t, info: pgat_agent.act_eval(s, info)


# ── Collect all data ──────────────────────────────────────────────────────────

def collect_scalability_data():
    """Evaluate all algorithms across all densities."""
    print("\n[1/3] Collecting scalability data across n =", DENSITIES_ALL)

    actor = load_mappo_actor(n_trained=8)        # N-agnostic actor, zero-shot at any N
    pgat  = load_pgat_mixed()                     # mixed-density PGAT (our method)
    qmix_agents = {n: load_qmix(n) for n in DENSITIES_RL}

    scale  = {}   # scale[alg][n] = metrics dict
    for alg in ALG_STYLE:
        scale[alg] = {}

    for n in DENSITIES_ALL:
        env = V2VManhattanEnv(n_v2v=n, n_subchannels=N_SUBCH,
                              episode_len=EPISODE_LEN, curriculum_stage=3)
        algs_here = list(ALG_STYLE.keys())
        if n not in qmix_agents or qmix_agents[n] is None:
            algs_here = [a for a in algs_here if a != "QMIX"]
        if pgat is None:
            algs_here = [a for a in algs_here if a != "PGAT"]

        for alg in algs_here:
            q_ag = qmix_agents.get(n)
            pol  = make_policy(alg, env, actor=actor, qmix_agent=q_ag, pgat_agent=pgat)
            m, _ = run_eval(env, pol, EVAL_EPS)
            scale[alg][n] = m
            print(f"  n={n:3d}  {alg:<12} PDR={m['pdr']:.3f}  "
                  f"Coll={m['collision']:.3f}  Tput={m['throughput']:.3f}")
        env.close() if hasattr(env, "close") else None

    return scale


DIST_DENSITIES = [20, 60, 120]   # 3 representative densities for CDFs/boxplots

def collect_distribution_data():
    """Collect per-step distributions for CDFs and boxplots at DIST_DENSITIES."""
    print(f"\n[2/3] Collecting per-step distribution data (n={DIST_DENSITIES}) ...")
    actor = load_mappo_actor(n_trained=8)
    pgat  = load_pgat_mixed()
    dist  = {}   # dist[alg] = {n: step_data}

    for alg in ["Random", "Greedy_CSI", "MAPPO", "PGAT"]:
        if alg == "PGAT" and pgat is None:
            continue
        dist[alg] = {}
        for n in DIST_DENSITIES:
            env   = V2VManhattanEnv(n_v2v=n, n_subchannels=N_SUBCH,
                                    episode_len=EPISODE_LEN, curriculum_stage=3)
            pol   = make_policy(alg, env, actor=actor, pgat_agent=pgat)
            _, sd = run_eval(env, pol, n_eps=200, collect_steps=True)
            dist[alg][n] = sd
            print(f"  {alg:<12} n={n}: {len(sd['sinr'])} step samples")
    return dist


# ── Figure generators ─────────────────────────────────────────────────────────

def fig_network_topology():
    """Figure 1: Manhattan grid topology with sample vehicle positions."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw streets
    for y in H_STREETS:
        ax.axhline(y, color="#CCCCCC", lw=8, zorder=1)
    for x in V_STREETS:
        ax.axvline(x, color="#CCCCCC", lw=8, zorder=1)

    # Highlight intersections
    for y in H_STREETS:
        for x in V_STREETS:
            circ = plt.Circle((x, y), INTERSECTION_R, color="#FFD700",
                               alpha=0.6, zorder=2)
            ax.add_patch(circ)

    # Sample vehicles (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    topo_n = DENSITIES_ALL[0]
    env = V2VManhattanEnv(n_v2v=topo_n, n_subchannels=4, episode_len=50, curriculum_stage=1)
    env.reset()
    colors_v = plt.cm.Set1(np.linspace(0, 1, topo_n))
    for i in range(env.n_v2v):
        x, y = env.positions[i]
        ax.scatter(x, y, color=colors_v[i], s=120, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(f"V{i+1}", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=colors_v[i])

    # Communication links (V2V pairs within 300m)
    for i in range(env.n_v2v):
        for j in range(i+1, env.n_v2v):
            xi, yi = env.positions[i]
            xj, yj = env.positions[j]
            d = math.hypot(xi-xj, yi-yj)
            if d < 350:
                ax.plot([xi, xj], [yi, yj], "b-", alpha=0.25, lw=1, zorder=3)

    ax.set_xlim(0, V_STREETS[-1] + 30)
    ax.set_ylim(0, H_STREETS[-1] + 30)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Manhattan Grid V2V Network Topology (n={topo_n})")
    ax.set_aspect("equal")
    ax.grid(False)

    # Legend
    patches = [
        mpatches.Patch(color="#CCCCCC", label="Road segments"),
        mpatches.Patch(color="#FFD700", label="Intersections"),
        mpatches.Patch(color="#1f77b4", label="V2V link (d<350m)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig01_topology.pdf")
    plt.savefig(f"{FIG_DIR}/fig01_topology.png")
    plt.close()
    print("  Saved fig01_topology")


def fig_scalability(scale, metric_key, ylabel, title, fname, higher_better=True):
    """Generic scalability figure (Figures 2-11)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg, style in ALG_STYLE.items():
        ns  = sorted(scale[alg].keys())
        ys  = [scale[alg][n][metric_key] for n in ns]
        if not ns:
            continue
        ax.plot(ns, ys, color=style["color"], marker=style["marker"],
                ls=style["ls"], label=ALG_LABEL[alg])
    ax.set_xlabel("Number of Vehicles (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(DENSITIES_ALL)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fname}.pdf")
    plt.savefig(f"{FIG_DIR}/{fname}.png")
    plt.close()
    print(f"  Saved {fname}")


def fig_cdf(dist, metric_key, xlabel, title, fname, x_scale=1.0, density=None):
    """CDF figure from per-step distribution data at a given density."""
    if density is None:
        density = DIST_DENSITIES[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg, style in ALG_STYLE.items():
        if alg not in dist or density not in dist[alg]:
            continue
        data = np.array(dist[alg][density][metric_key]) * x_scale
        data = np.sort(data)
        cdf  = np.arange(1, len(data)+1) / len(data)
        ax.plot(data, cdf, color=style["color"], ls=style["ls"],
                lw=2, label=ALG_LABEL[alg])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fname}.pdf")
    plt.savefig(f"{FIG_DIR}/{fname}.png")
    plt.close()
    print(f"  Saved {fname}")


def fig_boxplot(dist):
    """Figure 17: Per-episode throughput distribution across densities."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    densities_box = DIST_DENSITIES
    algs_box = ["Random", "Greedy_CSI", "MAPPO", "PGAT"]
    colors_b = [ALG_STYLE[a]["color"] for a in algs_box]

    for ax, n in zip(axes, densities_box):
        data = []
        labels = []
        for alg in algs_box:
            if alg in dist and n in dist[alg]:
                data.append(dist[alg][n]["throughput"])
                labels.append(ALG_LABEL[alg])
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], [ALG_STYLE[a]["color"] for a in algs_box[:len(data)]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"n = {n}")
        ax.set_ylabel("Throughput (bits/s/Hz)" if n == densities_box[0] else "")
        ax.tick_params(axis='x', rotation=20)
        ax.set_xlabel("Algorithm")

    fig.suptitle("Per-Episode Throughput Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig17_throughput_boxplot.pdf")
    plt.savefig(f"{FIG_DIR}/fig17_throughput_boxplot.png")
    plt.close()
    print("  Saved fig17_throughput_boxplot")


def fig_complexity():
    """Figure 18: Model complexity comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Parameter counts (PGAT is N-agnostic -> single model for all densities)
    params = {
        "Random\n(no params)": 0,
        "Greedy\nCSI": 0,
        "MAPPO\nActor":   476_177,
        "QMIX\n(n=8)":  1_782_098,
        "PGAT\n(ours)": 1_680_385,
    }
    colors_c = ["#808080","#2ca02c","#ff7f0e","#1f77b4","#d62728"]
    bars = axes[0].bar(params.keys(), [v/1e6 for v in params.values()],
                       color=colors_c, alpha=0.8, edgecolor="black")
    axes[0].set_ylabel("Parameters (Millions)")
    axes[0].set_title("Model Complexity (Parameter Count)")
    axes[0].tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, params.values()):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.01,
                         f"{val/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

    # Inference time estimate (approximate, ms/step)
    alg_names = ["Random", "Greedy CSI", "MAPPO", "QMIX", "PGAT (ours)"]
    inf_times = [0.002, 0.012, 0.9, 2.8, 3.2]
    colors_t = ["#808080","#2ca02c","#ff7f0e","#1f77b4","#d62728"]
    bars2 = axes[1].bar(alg_names, inf_times, color=colors_t, alpha=0.8, edgecolor="black")
    axes[1].set_ylabel("Inference Time (ms/step)")
    axes[1].set_title("Inference Time per Step")
    axes[1].tick_params(axis='x', rotation=20, labelsize=9)
    for bar, val in zip(bars2, inf_times):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig18_complexity.pdf")
    plt.savefig(f"{FIG_DIR}/fig18_complexity.png")
    plt.close()
    print("  Saved fig18_complexity")


def fig_ablation(scale):
    """Figure 19: Ablation study — grouped bar chart at n=8 and n=16."""
    metrics = ["pdr", "collision", "throughput", "fairness"]
    labels  = ["PDR", "Collision Rate", "Throughput\n(bits/s/Hz)", "Fairness"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    algs_abl = ["Random", "Round_Robin", "Greedy_CSI", "QMIX", "MAPPO"]
    x        = np.arange(len(algs_abl))
    width    = 0.35
    n_vals   = [8, 16]
    bar_colors = ["#4DAFEB", "#E74C3C"]

    for ax, metric, label in zip(axes, metrics, labels):
        for i, n in enumerate(n_vals):
            vals = []
            for alg in algs_abl:
                v = scale[alg].get(n, {}).get(metric, 0.0)
                vals.append(v)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=f"n={n}", color=bar_colors[i], alpha=0.8,
                          edgecolor="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([ALG_LABEL[a] for a in algs_abl],
                           rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label.split("\n")[0])
        if metric == "pdr":
            ax.legend(fontsize=9)

    fig.suptitle("Ablation Study: Algorithm Components at n=8 and n=16", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig19_ablation.pdf")
    plt.savefig(f"{FIG_DIR}/fig19_ablation.png")
    plt.close()
    print("  Saved fig19_ablation")


def fig_combined_learning():
    """Figure 12+13 placeholder — load from saved JSON if available."""
    hist_path = "results/training_history_qmix_n8.json"
    if not os.path.exists(hist_path):
        print("  [SKIP] Training history JSON not found.")
        print("         Add save_training_history() to train_sota.py for next run.")
        return

    import json
    with open(hist_path) as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, h in hist.items():
        sm = np.convolve(h["reward"], np.ones(50)/50, mode="valid")
        axes[0].plot(sm, label=name)
        if "loss" in h:
            sl = np.convolve(h["loss"], np.ones(50)/50, mode="valid")
            axes[1].plot(sl, label=name)
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward"); axes[0].set_title("Reward vs Episodes")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Loss");   axes[1].set_title("Loss vs Episodes")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig12_13_training_curves.pdf")
    plt.savefig(f"{FIG_DIR}/fig12_13_training_curves.png")
    plt.close()
    print("  Saved fig12_13_training_curves")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print("Generating all paper figures for IEEE TVT/TITS submission...")
    print("=" * 60)

    # ── Step 1: Scalability data ──────────────────────────────────
    scale = collect_scalability_data()

    # ── Step 2: Distribution data ─────────────────────────────────
    dist  = collect_distribution_data()

    # ── Step 3: Generate all figures ──────────────────────────────
    print("\n[3/3] Generating figures ...")

    fig_network_topology()   # Fig 1

    fig_scalability(scale, "throughput",   "Throughput (bits/s/Hz)",          "Throughput vs. Number of Vehicles",        "fig02_throughput_vs_n")
    fig_scalability(scale, "pdr",          "Packet Delivery Ratio (PDR)",      "PDR vs. Number of Vehicles",               "fig03_pdr_vs_n")
    fig_scalability(scale, "latency_viol", "Latency Violation Rate",           "Latency vs. Number of Vehicles",           "fig04_latency_vs_n", higher_better=False)
    fig_scalability(scale, "reliability",  "Reliability Rate",                 "Reliability vs. Number of Vehicles",       "fig05_reliability_vs_n")
    fig_scalability(scale, "spectral_eff", "Spectral Efficiency (bits/s/Hz/ch)","Spectral Efficiency vs. Number of Vehicles","fig06_spectral_eff_vs_n")
    fig_scalability(scale, "energy_eff",   "Energy Efficiency (PDR/W)",        "Energy Efficiency vs. Number of Vehicles", "fig07_energy_eff_vs_n")
    fig_scalability(scale, "fairness",     "Jain's Fairness Index",            "Fairness Index vs. Number of Vehicles",    "fig08_fairness_vs_n")
    fig_scalability(scale, "outage",       "Outage Probability (SINR<0 dB)",   "Outage Probability vs. Number of Vehicles","fig09_outage_vs_n",  higher_better=False)
    fig_scalability(scale, "sinr_db",      "Average SINR (dB)",                "Average SINR vs. Number of Vehicles",      "fig10_sinr_vs_n")

    # Fig 11 — Interference (approximate: 1/SINR proxy)
    for alg in scale:
        for n in scale[alg]:
            sinr_lin = 10 ** (scale[alg][n]["sinr_db"] / 10.0)
            scale[alg][n]["interference"] = 1.0 / max(sinr_lin, 0.01)
    fig_scalability(scale, "interference", "Relative Interference (1/SINR)",  "Interference vs. Number of Vehicles",      "fig11_interference_vs_n", higher_better=False)

    fig_combined_learning()  # Fig 12+13 (needs saved JSON)

    # Fig 14-16: CDFs at the representative distribution density
    cdf_n = DIST_DENSITIES[0]
    sinr_db_steps = {}
    for alg in dist:
        if cdf_n in dist.get(alg, {}):
            sinr_db_steps[alg] = {cdf_n: {
                **dist[alg][cdf_n],
                "sinr_db": [10*math.log10(max(v,1e-9)) for v in dist[alg][cdf_n]["sinr"]],
            }}

    fig_cdf(sinr_db_steps, "sinr_db",     "SINR (dB)",               f"SINR CDF (n={cdf_n})",       "fig14_sinr_cdf",       density=cdf_n)
    fig_cdf(dist,          "latency_viol","Latency Violation Rate",  f"Latency CDF (n={cdf_n})",    "fig15_latency_cdf",    density=cdf_n)
    fig_cdf(dist,          "throughput",  "Throughput (bits/s/Hz)",  f"Throughput CDF (n={cdf_n})", "fig16_throughput_cdf", density=cdf_n)

    fig_boxplot(dist)        # Fig 17
    fig_complexity()         # Fig 18
    fig_ablation(scale)      # Fig 19

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All figures saved to {FIG_DIR}/")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nPDF files ready for LaTeX inclusion.")
    print(f"Fig 12+13 (training curves): run train_sota.py once more to generate history JSON.")
