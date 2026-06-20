# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation - V2V Manhattan Grid MARL
===================================================
Evaluates all algorithms across vehicle densities n=8 to n=120.

Algorithm coverage per density:
  Baselines (Random, Greedy CSI, Round Robin) : ALL densities
  MAPPO actor (n=8 checkpoint)               : ALL densities  (actor is N-agnostic)
  QMIX                                        : trained densities only (needs N-specific mixer)

Metrics:
  PDR, Collision, Throughput, Net_Throughput, Spectral_Eff,
  Fairness, SINR_dB, Latency_Viol, Reliability, Energy_Eff, LoS_Ratio

Outputs:
  results/metrics_all_densities.csv   — full data (import into Excel / plot with matplotlib)
  results/metrics_summary.txt         — LaTeX-ready tables for paper
"""

import os
import csv
import math
import numpy as np
import torch
from collections import defaultdict

from env_manhattan import V2VManhattanEnv
from networks    import MAPPOActor
from agent_qmix  import QMIXAgent
from agent_graph_mappo import PGATMAPPOAgent
from baselines   import random_policy, greedy_csi_policy, round_robin_policy

os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_EPS     = 100      # episodes per (algorithm, density) — tight confidence interval
EPISODE_LEN  = 50
N_SUBCH      = 4
N_ACTIONS    = 16       # 4ch × 4pwr

# Target density sweep (PGAT trained on these; MAPPO evaluated zero-shot from n=8;
# QMIX auto-skips — its mixer is N-specific and no checkpoints exist at these N).
DENSITIES = [20, 40, 60, 80, 100, 120]

# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(acc: dict, n_v2v: int) -> dict:
    """Aggregate per-step info dicts into paper metrics."""
    pdr        = float(np.mean(acc["avg_pdr"]))
    collision  = float(np.mean(acc["avg_collision"]))
    tput       = float(np.mean(acc["avg_throughput"]))    # per-agent
    fairness   = float(np.mean(acc["fairness"]))
    sinr_lin   = float(np.mean(acc["avg_sinr"]))
    energy     = float(np.mean(acc["avg_energy"]))
    lat_viol   = float(np.mean(acc["latency_violation_rate"]))
    reliability= float(np.mean(acc["reliability_rate"]))
    los_ratio  = float(np.mean(acc.get("avg_los_ratio", [0.0])))

    sinr_db       = 10.0 * math.log10(max(sinr_lin, 1e-9))
    net_tput      = tput * n_v2v                          # total network throughput
    spectral_eff  = tput / N_SUBCH                        # bits/s/Hz per channel
    energy_eff    = pdr  / max(energy, 1e-9)              # PDR per unit energy

    return {
        "PDR":           round(pdr,        4),
        "Collision":     round(collision,  4),
        "Throughput":    round(tput,       4),
        "Net_Throughput":round(net_tput,   3),
        "Spectral_Eff":  round(spectral_eff,4),
        "Fairness":      round(fairness,   4),
        "SINR_dB":       round(sinr_db,    3),
        "Latency_Viol":  round(lat_viol,   4),
        "Reliability":   round(reliability,4),
        "Energy_Eff":    round(energy_eff, 4),
        "LoS_Ratio":     round(los_ratio,  4),
    }


def run_episodes(env, policy_fn, n_eps: int) -> dict:
    """
    Run n_eps episodes, return aggregated info.
    policy_fn signature: (state, step, info) -> actions.
    `info` carries the graph data PGAT needs; other policies ignore it.
    """
    acc = defaultdict(list)
    for _ in range(n_eps):
        state, info = env.reset()           # reset info already holds graph fields
        done = False
        step = 0
        while not done:
            actions = policy_fn(state, step, info)
            state, _, done, _, info = env.step(actions)
            step += 1
        for k, v in info.items():
            if isinstance(v, (int, float)):  # skip graph arrays (positions, etc.)
                acc[k].append(v)
    return acc


# ── Policy wrappers (all accept (state, step, info); non-PGAT ignore info) ─────

def wrap_baseline(fn, env):
    if fn is round_robin_policy:
        return lambda s, t, info: round_robin_policy(env, s, t)
    return lambda s, t, info: fn(env, s)


def load_mappo_actor(n_v2v_trained: int = 8):
    """Load MAPPO actor. Actor is N-agnostic: 32→16, works for any density."""
    tag  = "manhattan"
    path = f"models_sota/mappo_best_{tag}_n{n_v2v_trained}.pth"
    if not os.path.exists(path):
        print(f"  [MAPPO] checkpoint not found: {path}")
        return None
    ckpt  = torch.load(path, map_location=DEVICE)
    actor = MAPPOActor(state_dim=32, n_actions=N_ACTIONS, d_model=256).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"  [MAPPO] loaded actor from {path}  (trained n={n_v2v_trained}, eval any n)")
    return actor


def mappo_policy(actor: MAPPOActor):
    """Greedy deterministic policy using MAPPO actor (mode of Categorical)."""
    @torch.no_grad()
    def _act(state, _step, _info):
        s = torch.FloatTensor(state).to(DEVICE)
        dist = actor.get_dist(s)
        # Use mode (argmax of logits) for deterministic eval
        return dist.logits.argmax(dim=-1).cpu().numpy()
    return _act


def load_pgat_mixed():
    """Load the mixed-density PGAT model (N-agnostic, works at any density)."""
    for path in ("models_pgat/pgat_mixed_best.pth",
                 "models_pgat/pgat_mixed_final.pth"):
        if os.path.exists(path):
            ckpt  = torch.load(path, map_location=DEVICE)
            agent = PGATMAPPOAgent(state_dim=32, n_actions=N_ACTIONS, n_agents=20,
                                   d_model=128, n_heads=4, n_layers=3, device=DEVICE)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.actor.eval()
            agent.set_eval_temperature(0.3)   # argmax collapses all agents to same channel
            print(f"  [PGAT] loaded {path}  (mixed-density, eval any n, T=0.3)")
            return agent
    print("  [PGAT] no mixed checkpoint found in models_pgat/")
    return None


def pgat_policy(agent):
    """PGAT needs the env info dict (graph data) at each step."""
    return lambda s, _t, info: agent.act_eval(s, info)


def load_qmix_agent(n_v2v: int) -> QMIXAgent | None:
    """Load QMIX best checkpoint if it exists for this exact density."""
    path = f"models_sota/qmix_best_manhattan_n{n_v2v}.pth"
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    agent = QMIXAgent(
        state_dim=32, n_actions=N_ACTIONS, n_agents=n_v2v,
        d_model=256, n_heads=8, n_layers=3, d_ff=512,
        mixing_embed_dim=64, device=DEVICE,
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.q_network.eval()
    print(f"  [QMIX] loaded checkpoint: {path}")
    return agent


def qmix_policy(agent: QMIXAgent):
    return lambda s, _t, _info: agent.act(s, explore=False)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_all():
    print(f"\nDevice: {DEVICE}")
    print(f"Episodes per (algorithm × density): {EVAL_EPS}")
    print(f"Densities: {DENSITIES}\n")

    # Load models once. MAPPO actor is N-agnostic -> load the n=8 checkpoint and
    # evaluate it zero-shot at every density. PGAT mixed model handles any N too.
    mappo_actor = load_mappo_actor(n_v2v_trained=8)
    pgat_agent  = load_pgat_mixed()

    all_rows = []   # for CSV
    summary  = {}   # density → {algorithm → metrics}

    for n in DENSITIES:
        print(f"\n{'─'*60}")
        print(f"  n_v2v = {n}")
        print(f"{'─'*60}")
        summary[n] = {}

        env = V2VManhattanEnv(
            n_v2v=n, n_subchannels=N_SUBCH,
            episode_len=EPISODE_LEN, curriculum_stage=3
        )

        algorithms = {
            "Random":      wrap_baseline(random_policy,      env),
            "Greedy_CSI":  wrap_baseline(greedy_csi_policy,  env),
            "Round_Robin": wrap_baseline(round_robin_policy, env),
        }

        if mappo_actor is not None:
            algorithms["MAPPO"] = mappo_policy(mappo_actor)

        # QMIX only if exact checkpoint exists for this density (N-specific mixer)
        qmix_agent = load_qmix_agent(n)
        if qmix_agent is not None:
            algorithms["QMIX"] = qmix_policy(qmix_agent)

        # PGAT mixed-density model (N-agnostic) — our method
        if pgat_agent is not None:
            algorithms["PGAT"] = pgat_policy(pgat_agent)

        for alg_name, policy_fn in algorithms.items():
            print(f"  {alg_name:<14} ...", end="", flush=True)
            acc = run_episodes(env, policy_fn, EVAL_EPS)
            m   = compute_metrics(acc, n)
            summary[n][alg_name] = m
            print(f" PDR={m['PDR']:.3f}  Coll={m['Collision']:.3f}"
                  f"  Tput={m['Throughput']:.3f}  SINR={m['SINR_dB']:.1f}dB"
                  f"  Fair={m['Fairness']:.3f}  Rel={m['Reliability']:.3f}")
            row = {"n_v2v": n, "Algorithm": alg_name, **m}
            all_rows.append(row)

    return all_rows, summary


# ── Output helpers ────────────────────────────────────────────────────────────

METRIC_COLS = [
    "PDR", "Collision", "Throughput", "Net_Throughput",
    "Spectral_Eff", "Fairness", "SINR_dB",
    "Latency_Viol", "Reliability", "Energy_Eff", "LoS_Ratio",
]

def save_csv(rows: list, path: str):
    fieldnames = ["n_v2v", "Algorithm"] + METRIC_COLS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV saved → {path}")


def save_summary_txt(summary: dict, path: str):
    lines = []

    def bar(n=72): return "═" * n

    lines += [
        bar(),
        "  V2V MARL — COMPREHENSIVE EVALUATION RESULTS",
        "  Scenario : Manhattan Grid  |  100 episodes per cell  |  Stage-3 env",
        bar(),
        "",
    ]

    # ── Table 1: PDR across densities ──────────────────────────────────────
    _ORDER = ["Random", "Greedy_CSI", "Round_Robin", "MAPPO", "QMIX", "PGAT"]
    algs_present = sorted({a for n in summary for a in summary[n]},
                          key=lambda x: _ORDER.index(x) if x in _ORDER else 99)

    lines.append("  TABLE 1: Packet Delivery Ratio (PDR) by Density")
    lines.append("  " + "─" * 68)
    header = f"  {'Algorithm':<14}" + "".join(f"  n={n:>3}" for n in DENSITIES)
    lines.append(header)
    lines.append("  " + "─" * 68)
    for alg in algs_present:
        row = f"  {alg:<14}"
        for n in DENSITIES:
            val = summary[n].get(alg, {}).get("PDR", None)
            row += f"  {val:.3f}" if val is not None else "      —"
        lines.append(row)
    lines.append("")

    # ── Table 2: Collision Rate ─────────────────────────────────────────────
    lines.append("  TABLE 2: Collision Rate (lower is better)")
    lines.append("  " + "─" * 68)
    lines.append(header)
    lines.append("  " + "─" * 68)
    for alg in algs_present:
        row = f"  {alg:<14}"
        for n in DENSITIES:
            val = summary[n].get(alg, {}).get("Collision", None)
            row += f"  {val:.3f}" if val is not None else "      —"
        lines.append(row)
    lines.append("")

    # ── Table 3: Throughput ─────────────────────────────────────────────────
    lines.append("  TABLE 3: Per-Agent Throughput (bits/s/Hz)")
    lines.append("  " + "─" * 68)
    lines.append(header)
    lines.append("  " + "─" * 68)
    for alg in algs_present:
        row = f"  {alg:<14}"
        for n in DENSITIES:
            val = summary[n].get(alg, {}).get("Throughput", None)
            row += f"  {val:.3f}" if val is not None else "      —"
        lines.append(row)
    lines.append("")

    # ── Table 4: Fairness ───────────────────────────────────────────────────
    lines.append("  TABLE 4: Jain's Fairness Index")
    lines.append("  " + "─" * 68)
    lines.append(header)
    lines.append("  " + "─" * 68)
    for alg in algs_present:
        row = f"  {alg:<14}"
        for n in DENSITIES:
            val = summary[n].get(alg, {}).get("Fairness", None)
            row += f"  {val:.3f}" if val is not None else "      —"
        lines.append(row)
    lines.append("")

    # ── Table 5: Full metric table at the lowest swept density ─────────────
    _n0 = DENSITIES[0]
    lines.append(f"  TABLE 5: Full Metrics at n={_n0} (Lowest Swept Density)")
    lines.append("  " + "─" * 68)
    col_w = 10
    hdr   = f"  {'Metric':<18}" + "".join(f"{a:>{col_w}}" for a in algs_present)
    lines.append(hdr)
    lines.append("  " + "─" * 68)
    for metric in METRIC_COLS:
        row = f"  {metric:<18}"
        for alg in algs_present:
            val = summary[_n0].get(alg, {}).get(metric, None)
            row += f"{val:{col_w}.4f}" if val is not None else " " * col_w + "—"
        lines.append(row)
    lines.append("")

    # ── MAPPO scalability gain vs Random ───────────────────────────────────
    if "MAPPO" in algs_present:
        lines.append("  TABLE 6: MAPPO PDR Gain vs Random Baseline (scalability)")
        lines.append("  " + "─" * 52)
        lines.append(f"  {'n_v2v':<8}  {'Random PDR':>12}  {'MAPPO PDR':>12}  {'Gain':>8}")
        lines.append("  " + "─" * 52)
        for n in DENSITIES:
            r = summary[n].get("Random", {}).get("PDR", None)
            m = summary[n].get("MAPPO",  {}).get("PDR", None)
            if r and m:
                gain = (m - r) / max(r, 1e-9) * 100
                lines.append(f"  {n:<8}  {r:>12.3f}  {m:>12.3f}  {gain:>7.1f}%")
        lines.append("")

    lines.append(bar())

    txt = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  Summary saved → {path}")
    print()
    print(txt)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rows, summary = evaluate_all()
    save_csv(rows,    "results/metrics_all_densities.csv")
    save_summary_txt(summary, "results/metrics_summary.txt")
    print("\nDone. Use the CSV for plotting and metrics_summary.txt for paper tables.")
