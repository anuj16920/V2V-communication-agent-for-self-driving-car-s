"""
Re-evaluate saved best checkpoints to get true peak performance.
Run: python eval_best.py
"""
import numpy as np
import torch
from collections import defaultdict

from env_manhattan import V2VManhattanEnv
from agent_qmix  import QMIXAgent
from agent_mappo import MAPPOAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_EPS = 100   # more episodes → tighter estimate


def evaluate(env, policy_fn, episodes=100):
    acc = defaultdict(list)
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            actions = policy_fn(state)
            state, _, done, _, info = env.step(actions)
        for k, v in info.items():
            acc[k].append(v)
    return {
        "PDR":        float(np.mean(acc["avg_pdr"])),
        "Collision":  float(np.mean(acc["avg_collision"])),
        "Throughput": float(np.mean(acc["avg_throughput"])),
        "Fairness":   float(np.mean(acc["fairness"])),
    }


def eval_qmix_best(n_v2v=8):
    path = f"models_sota/qmix_best_manhattan_n{n_v2v}.pth"
    ckpt = torch.load(path, map_location=DEVICE)

    env = V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
    agent = QMIXAgent(
        state_dim=env.STATE_DIM, n_actions=env.n_actions, n_agents=n_v2v,
        d_model=256, n_heads=8, n_layers=3, d_ff=512,
        mixing_embed_dim=64, device=DEVICE,
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.q_network.eval()

    res = evaluate(env, lambda s: agent.act(s, explore=False), EVAL_EPS)
    print(f"  QMIX best  (n={n_v2v}): PDR={res['PDR']:.3f}  Coll={res['Collision']:.3f}"
          f"  Tput={res['Throughput']:.3f}  Fair={res['Fairness']:.3f}")
    return res


def eval_mappo_best(n_v2v=8):
    path = f"models_sota/mappo_best_manhattan_n{n_v2v}.pth"
    ckpt = torch.load(path, map_location=DEVICE)

    env = V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
    agent = MAPPOAgent(
        state_dim=env.STATE_DIM, n_actions=env.n_actions, n_agents=n_v2v,
        d_model=256, device=DEVICE,
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()

    @torch.no_grad()
    def policy(state):
        s = torch.FloatTensor(state).to(agent.device)
        return agent.actor.get_dist(s).mode.cpu().numpy()  # deterministic greedy

    res = evaluate(env, policy, EVAL_EPS)
    print(f"  MAPPO best (n={n_v2v}): PDR={res['PDR']:.3f}  Coll={res['Collision']:.3f}"
          f"  Tput={res['Throughput']:.3f}  Fair={res['Fairness']:.3f}")
    return res


def eval_baselines(n_v2v=8):
    from baselines import random_policy, greedy_csi_policy, round_robin_policy
    env = V2VManhattanEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
    for name, fn in [("Random", random_policy), ("Greedy CSI", greedy_csi_policy),
                     ("Round Robin", round_robin_policy)]:
        res = evaluate(env, lambda s, _fn=fn: _fn(env, s), EVAL_EPS)
        print(f"  {name:<12} (n={n_v2v}): PDR={res['PDR']:.3f}  Coll={res['Collision']:.3f}"
              f"  Tput={res['Throughput']:.3f}  Fair={res['Fairness']:.3f}")


if __name__ == "__main__":
    print(f"\nDevice: {DEVICE}")
    print("\n── Baselines ──────────────────────────────────────────────")
    eval_baselines(n_v2v=8)
    print("\n── Best checkpoints ───────────────────────────────────────")
    eval_qmix_best(n_v2v=8)
    eval_mappo_best(n_v2v=8)
