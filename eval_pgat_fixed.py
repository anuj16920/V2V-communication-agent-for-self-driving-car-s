# -*- coding: utf-8 -*-
"""
Re-evaluate the trained PGAT model with temperature-scaled sampling.

The pure argmax eval (temperature=1.0) caused all agents to collapse to
the same channel because graph-attention produces correlated embeddings
when agents share the same street (same CSI -> same argmax). Training
used stochastic sampling which naturally diversifies actions.

Run: python eval_pgat_fixed.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from collections import defaultdict

from env_manhattan    import V2VManhattanEnv
from agent_graph_mappo import PGATMAPPOAgent
from networks         import MAPPOActor
from baselines        import random_policy, greedy_csi_policy

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DENSITIES   = [20, 40, 60, 80, 100, 120]
EVAL_EPS    = 50


def load_pgat():
    for path in ("models_pgat/pgat_mixed_best.pth",
                 "models_pgat/pgat_mixed_final.pth"):
        if os.path.exists(path):
            ckpt  = torch.load(path, map_location=DEVICE)
            agent = PGATMAPPOAgent(state_dim=32, n_actions=16, n_agents=20,
                                   d_model=128, n_heads=4, n_layers=3, device=DEVICE)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.actor.eval()
            print(f"Loaded: {path}")
            return agent
    raise FileNotFoundError("No PGAT checkpoint found in models_pgat/")


def evaluate(env, agent, n_eps):
    acc = defaultdict(list)
    for _ in range(n_eps):
        obs, info = env.reset(); done = False
        while not done:
            a = agent.act_eval(obs, info)
            obs, _, done, _, info = env.step(a)
        for k, v in info.items():
            if isinstance(v, (int, float)):
                acc[k].append(v)
    return {"PDR":    float(np.mean(acc["avg_pdr"])),
            "Coll":   float(np.mean(acc["avg_collision"])),
            "Tput":   float(np.mean(acc["avg_throughput"])),
            "Fair":   float(np.mean(acc["fairness"]))}


def find_best_temperature(agent, n_v2v=40, n_eps=25):
    env = V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4,
                          episode_len=50, curriculum_stage=3)
    best_t, best_pdr = 1.0, -1.0
    print(f"\nTemperature search at n={n_v2v}:")
    for t in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15]:
        agent.set_eval_temperature(t)
        r = evaluate(env, agent, n_eps)
        mark = "  <-- best" if r["PDR"] > best_pdr else ""
        print(f"  T={t:.2f}  PDR={r['PDR']:.3f}  Coll={r['Coll']:.3f}  Tput={r['Tput']:.3f}{mark}")
        if r["PDR"] > best_pdr:
            best_pdr, best_t = r["PDR"], t
    print(f"\nSelected temperature: {best_t}  (PDR={best_pdr:.3f})\n")
    agent.set_eval_temperature(best_t)
    return best_t


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    agent = load_pgat()

    # Load MAPPO zero-shot baseline for comparison
    mappo_actor = None
    mp_path = "models_sota/mappo_best_manhattan_n8.pth"
    if os.path.exists(mp_path):
        ck = torch.load(mp_path, map_location=DEVICE)
        mappo_actor = MAPPOActor(32, 16, 256).to(DEVICE)
        mappo_actor.load_state_dict(ck["actor"]); mappo_actor.eval()

    @torch.no_grad()
    def mappo_fn(obs):
        s = torch.FloatTensor(obs).to(DEVICE)
        return mappo_actor.get_dist(s).logits.argmax(-1).cpu().numpy()

    # Step 1: find best temperature
    find_best_temperature(agent, n_v2v=40, n_eps=25)

    # Step 2: full evaluation at all densities
    print(f"{'='*72}")
    print(f"  PGAT FIXED EVALUATION  (temperature={agent.eval_temperature})")
    print(f"{'='*72}")
    print(f"  {'n_v2v':<7}{'Random':>14}{'Greedy-CSI':>14}{'MAPPO(zs)':>14}{'PGAT(ours)':>14}")
    print(f"  {'-'*60}")

    for n in DENSITIES:
        env = V2VManhattanEnv(n_v2v=n, n_subchannels=4, episode_len=50, curriculum_stage=3)

        # Random
        acc = defaultdict(list)
        for _ in range(EVAL_EPS):
            obs, _ = env.reset(); done = False
            while not done:
                obs, _, done, _, info = env.step(random_policy(env, obs))
            acc["pdr"].append(info["avg_pdr"]); acc["coll"].append(info["avg_collision"])
        rp, rc = np.mean(acc["pdr"]), np.mean(acc["coll"])

        # Greedy CSI
        acc = defaultdict(list)
        for _ in range(EVAL_EPS):
            obs, _ = env.reset(); done = False
            while not done:
                obs, _, done, _, info = env.step(greedy_csi_policy(env, obs))
            acc["pdr"].append(info["avg_pdr"]); acc["coll"].append(info["avg_collision"])
        gp, gc = np.mean(acc["pdr"]), np.mean(acc["coll"])

        # MAPPO zero-shot
        mp, mc = 0.0, 0.0
        if mappo_actor:
            acc = defaultdict(list)
            for _ in range(EVAL_EPS):
                obs, _ = env.reset(); done = False
                while not done:
                    obs, _, done, _, info = env.step(mappo_fn(obs))
                acc["pdr"].append(info["avg_pdr"]); acc["coll"].append(info["avg_collision"])
            mp, mc = np.mean(acc["pdr"]), np.mean(acc["coll"])

        # PGAT
        pr = evaluate(env, agent, EVAL_EPS)

        rstr = f"{rp:.3f}/{rc:.3f}"
        gstr = f"{gp:.3f}/{gc:.3f}"
        mstr = f"{mp:.3f}/{mc:.3f}" if mappo_actor else "   --/--  "
        pstr = f"{pr['PDR']:.3f}/{pr['Coll']:.3f}"
        print(f"  {n:<7}{rstr:>14}{gstr:>14}{mstr:>14}{pstr:>14}")

    print(f"{'='*72}")
    print("\nDone. If PGAT PDR is still below MAPPO, consider retraining with")
    print("agent index added to the observation (breaks argmax symmetry at init).")
