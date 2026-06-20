# -*- coding: utf-8 -*-
"""
Confusion Matrix for V2V MARL Algorithms
==========================================
Compares each algorithm's per-step delivery outcome against
the Greedy CSI oracle (best achievable without learning).

Definition:
  Positive (1) = SINR >= threshold  -> reliable packet delivery
  Negative (0) = SINR <  threshold  -> packet lost / degraded

For each algorithm we compute:
  TP = Both algorithm AND oracle achieved reliable delivery
  FP = Algorithm reliable, oracle was NOT (algorithm got lucky)
  FN = Oracle reliable, algorithm was NOT (algorithm missed)
  TN = Both failed (step was inherently bad, high interference)

Derived metrics: Precision, Recall, F1, Accuracy

Outputs: results/figures/confusion_matrices.png  (4 side-by-side heatmaps)
         results/figures/classification_metrics.png (bar chart)
"""

import os, math, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
warnings.filterwarnings("ignore")

from env_manhattan import V2VManhattanEnv
from networks      import MAPPOActor
from agent_qmix    import QMIXAgent
from baselines     import random_policy, greedy_csi_policy, round_robin_policy

os.makedirs("results/figures", exist_ok=True)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SINR_THRESH = 1.0      # 0 dB -- minimum SINR for reliable V2V delivery
EVAL_EPS    = 200      # episodes (more = tighter estimate)
N_V2V       = 8        # evaluate at trained density
EPISODE_LEN = 50

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Load models ───────────────────────────────────────────────────────────────

def load_mappo():
    ckpt  = torch.load(f"models_sota/mappo_best_manhattan_n{N_V2V}.pth", map_location=DEVICE)
    actor = MAPPOActor(state_dim=32, n_actions=16, d_model=256).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor

def load_qmix():
    ckpt  = torch.load(f"models_sota/qmix_best_manhattan_n{N_V2V}.pth", map_location=DEVICE)
    agent = QMIXAgent(state_dim=32, n_actions=16, n_agents=N_V2V,
                      d_model=256, n_heads=8, n_layers=3, d_ff=512,
                      mixing_embed_dim=64, device=DEVICE)
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.q_network.eval()
    return agent

@torch.no_grad()
def mappo_act(actor, state):
    s = torch.FloatTensor(state).to(DEVICE)
    return actor.get_dist(s).logits.argmax(dim=-1).cpu().numpy()

# ── Collect per-step SINR labels ──────────────────────────────────────────────

def collect_sinr_labels(env, policy_fn, n_eps):
    """Returns binary array: 1 if avg_sinr >= SINR_THRESH, else 0."""
    labels = []
    for _ in range(n_eps):
        state, _ = env.reset()
        done = False; t = 0
        while not done:
            actions      = policy_fn(state, t)
            state, _, done, _, info = env.step(actions)
            sinr_lin     = info["avg_sinr"]
            labels.append(1 if sinr_lin >= SINR_THRESH else 0)
            t += 1
    return np.array(labels, dtype=np.int32)

# ── Compute confusion matrix ──────────────────────────────────────────────────

def confusion_matrix_2x2(y_true, y_pred):
    """
    y_true = oracle labels, y_pred = algorithm labels
    Returns [[TN, FP], [FN, TP]] as 2x2 numpy array
    """
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    return np.array([[TN, FP], [FN, TP]]), TP, FP, FN, TN

def classification_metrics(TP, FP, FN, TN):
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / max(total, 1)
    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(Accuracy=accuracy, Precision=precision, Recall=recall, F1=f1)

# ── Plot functions ────────────────────────────────────────────────────────────

def plot_confusion_matrices(cms, alg_names, oracle_name="Greedy CSI"):
    n     = len(alg_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    tick_labels = ["Unreliable\n(SINR<0dB)", "Reliable\n(SINR>=0dB)"]
    colors_map  = ["#2196F3", "#F44336"]   # Blue=low, Red=high

    for ax, (cm, alg) in zip(axes, zip(cms, alg_names)):
        total = cm.sum()
        # Normalize to percentages
        cm_pct = cm / max(total, 1) * 100.0

        sns.heatmap(cm_pct, annot=False, fmt=".1f", ax=ax,
                    cmap="Blues", linewidths=0.5, linecolor="gray",
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    vmin=0, vmax=60, cbar=True)

        # Annotate each cell with count + percentage
        for i in range(2):
            for j in range(2):
                count = int(cm[i, j])
                pct   = cm_pct[i, j]
                cell_labels = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}
                color = "white" if pct > 30 else "black"
                ax.text(j + 0.5, i + 0.35, f"{cell_labels[(i,j)]}",
                        ha="center", va="center", fontsize=13,
                        fontweight="bold", color=color)
                ax.text(j + 0.5, i + 0.65, f"{count:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, color=color)

        ax.set_title(f"{alg}\nvs {oracle_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Algorithm: {alg}", fontsize=10)
        ax.set_ylabel(f"Oracle: {oracle_name}", fontsize=10)

    fig.suptitle(
        f"Confusion Matrices: Reliable Delivery Classification\n"
        f"(n={N_V2V} vehicles, SINR threshold = {SINR_THRESH:.0f} = 0 dB, {EVAL_EPS} episodes)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrices.png")
    plt.savefig("results/figures/confusion_matrices.pdf")
    plt.close()
    print("  Saved confusion_matrices.png / .pdf")


def plot_classification_metrics(metrics_dict, oracle_name="Greedy CSI"):
    algs    = [a for a in metrics_dict if a != oracle_name]
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    x    = np.arange(len(algs))
    w    = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [metrics_dict[a][metric] for a in algs]
        bars = ax.bar(x + (i - 1.5) * w, vals, w,
                      label=metric, color=color, alpha=0.85, edgecolor="black", lw=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(algs, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Classification Metrics: Reliable Delivery Prediction\n"
        f"(Positive = SINR >= 0 dB, oracle = {oracle_name})"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/classification_metrics.png")
    plt.savefig("results/figures/classification_metrics.pdf")
    plt.close()
    print("  Saved classification_metrics.png / .pdf")


def plot_sinr_distribution(labels_dict):
    """Stacked bar showing Reliable vs Unreliable fraction per algorithm."""
    algs  = list(labels_dict.keys())
    rel   = [float(np.mean(labels_dict[a])) for a in algs]
    unrel = [1.0 - r for r in rel]

    fig, ax = plt.subplots(figsize=(8, 4))
    alg_colors = {
        "Random":      "#808080",
        "Round Robin": "#9467bd",
        "Greedy CSI":  "#2ca02c",
        "QMIX":        "#1f77b4",
        "MAPPO":       "#d62728",
    }
    colors = [alg_colors.get(a, "#333333") for a in algs]

    bars1 = ax.bar(algs, rel,   color=colors, alpha=0.85,
                   edgecolor="black", lw=0.8, label="Reliable (SINR >= 0 dB)")
    ax.bar(algs, unrel, bottom=rel, color=colors, alpha=0.3,
           edgecolor="black", lw=0.8, hatch="///", label="Unreliable (SINR < 0 dB)")

    for bar, v in zip(bars1, rel):
        ax.text(bar.get_x() + bar.get_width()/2, v/2,
                f"{v*100:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

    ax.set_ylabel("Fraction of Steps")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Reliable vs Unreliable Delivery Fraction (n={N_V2V} vehicles)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/reliability_fraction.png")
    plt.savefig("results/figures/reliability_fraction.pdf")
    plt.close()
    print("  Saved reliability_fraction.png / .pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Collecting {EVAL_EPS} episodes at n={N_V2V} ...\n")

    env    = V2VManhattanEnv(n_v2v=N_V2V, n_subchannels=4,
                             episode_len=EPISODE_LEN, curriculum_stage=3)
    actor  = load_mappo()
    qmix   = load_qmix()

    policies = {
        "Random":      lambda s, t: random_policy(env, s),
        "Round Robin": lambda s, t: round_robin_policy(env, s, t),
        "Greedy CSI":  lambda s, t: greedy_csi_policy(env, s),
        "QMIX":        lambda s, t: qmix.act(s, explore=False),
        "MAPPO":       lambda s, t: mappo_act(actor, s),
    }

    print("Collecting per-step SINR labels ...")
    labels = {}
    for name, pol in policies.items():
        labels[name] = collect_sinr_labels(env, pol, EVAL_EPS)
        rel = np.mean(labels[name]) * 100
        print(f"  {name:<14}: reliable {rel:.1f}%  "
              f"({int(np.sum(labels[name]))} / {len(labels[name])} steps)")

    # Confusion matrices vs Greedy CSI oracle
    oracle       = labels["Greedy CSI"]
    alg_names    = ["Random", "Round Robin", "QMIX", "MAPPO"]
    cms, metrics = [], {}

    print("\nConfusion matrix vs Greedy CSI oracle:")
    print(f"  {'Algorithm':<14}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'TP':>7}  {'FP':>7}  {'FN':>7}  {'TN':>7}")
    print("  " + "-" * 78)
    for alg in alg_names:
        cm, TP, FP, FN, TN = confusion_matrix_2x2(oracle, labels[alg])
        m = classification_metrics(TP, FP, FN, TN)
        metrics[alg] = m
        cms.append(cm)
        print(f"  {alg:<14}  {m['Accuracy']:>9.3f}  {m['Precision']:>9.3f}  "
              f"{m['Recall']:>9.3f}  {m['F1']:>9.3f}  "
              f"{TP:>7}  {FP:>7}  {FN:>7}  {TN:>7}")

    # Save results text
    os.makedirs("results", exist_ok=True)
    with open("results/confusion_matrix_results.txt", "w") as f:
        f.write(f"Confusion Matrix Results  (n={N_V2V}, threshold=SINR>={SINR_THRESH:.0f}, eps={EVAL_EPS})\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"  {'Algorithm':<14}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}\n")
        f.write("  " + "-" * 55 + "\n")
        for alg, m in metrics.items():
            f.write(f"  {alg:<14}  {m['Accuracy']:>9.3f}  {m['Precision']:>9.3f}  "
                    f"{m['Recall']:>9.3f}  {m['F1']:>9.3f}\n")
    print("\n  Saved results/confusion_matrix_results.txt")

    print("\nGenerating plots ...")
    plot_confusion_matrices(cms, alg_names)
    plot_classification_metrics(metrics)
    plot_sinr_distribution(labels)

    print("\nDone. Files saved to results/figures/")
    print("  - confusion_matrices.png     (4 heatmaps)")
    print("  - classification_metrics.png (grouped bar chart)")
    print("  - reliability_fraction.png   (stacked bar)")
