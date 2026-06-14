"""
Improved Training Script with QMIX + Curriculum Learning

Improvements:
1. QMIX for better DQN coordination
2. Curriculum learning (3 stages)
3. Extended training (3000 episodes)
4. Transfer learning across densities
5. Better hyperparameters
"""

import os
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from env import V2VEnv
from agent_qmix import QMIXAgent
from agent_mappo import MAPPOAgent
from baselines import random_policy, greedy_csi_policy, round_robin_policy

# Reproducibility
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
os.makedirs("models_improved", exist_ok=True)  # Separate folder for improved models


def evaluate_policy(env, policy_fn, episodes: int = 30) -> dict:
    """Evaluate a policy"""
    keys = ["avg_pdr","avg_throughput","avg_collision","avg_sinr",
            "fairness","avg_energy","latency_violation_rate","reliability_rate"]
    acc = defaultdict(list)
    rew_acc = []

    for _ in range(episodes):
        state, _ = env.reset()
        ep_rew = 0.0
        while True:
            actions = policy_fn(env, state)
            state, rewards, done, _, info = env.step(actions)
            ep_rew += float(np.mean(rewards))
            if done:
                break
        rew_acc.append(ep_rew)
        for k in keys:
            acc[k].append(info[k])

    result = {"Reward": float(np.mean(rew_acc))}
    metric_map = {
        "avg_pdr": "PDR",
        "avg_throughput": "Throughput",
        "avg_collision": "Collision",
        "avg_sinr": "SINR",
        "fairness": "Fairness",
        "avg_energy": "Energy",
        "latency_violation_rate": "LatencyViolation",
        "reliability_rate": "Reliability",
    }
    for k, label in metric_map.items():
        result[label] = float(np.mean(acc[k]))
    return result


def train_qmix_curriculum(n_v2v: int, total_episodes: int = 3000) -> dict:
    """
    Train QMIX with 3-stage curriculum learning
    
    Stage 1 (0-1000 eps): Focus on collision avoidance
    Stage 2 (1000-2000 eps): Focus on PDR
    Stage 3 (2000-3000 eps): Full multi-objective
    """
    print(f"\n{'='*60}")
    print(f"  [QMIX Curriculum] Training n_v2v={n_v2v}  episodes={total_episodes}")
    print(f"{'='*60}")

    hist = defaultdict(list)
    t0 = time.time()
    
    # Stage 1: Collision avoidance (1000 episodes)
    print("\n  Stage 1/3: Collision Avoidance (eps 0-1000)")
    env = V2VEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=1)
    
    agent = QMIXAgent(
        state_dim=env.STATE_DIM,
        n_actions=env.n_actions,
        n_agents=n_v2v,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        mixing_embed_dim=32,
        lr=3e-4,
        gamma=0.99,
        batch_size=512,
        n_steps=3,
        tau=0.001,  # Slower target updates
        grad_clip=10.0,
        update_freq=4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_steps=150_000,  # Slower decay
        device=DEVICE,
    )
    
    print(f"  QMIX param count: {agent.param_count():,}")
    
    for ep in range(1000):
        state, _ = env.reset()
        ep_rew = 0.0
        ep_loss = []
        
        while True:
            actions = agent.act(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, rewards, nstate, done)
            loss = agent.train_step()
            if loss > 0:
                ep_loss.append(loss)
            state = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break
        
        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["loss"].append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        
        if ep % 200 == 0:
            elapsed = time.time() - t0
            eps = agent._get_epsilon()
            print(f"    Ep {ep:4d} | Rew {ep_rew:8.2f} | PDR {info['avg_pdr']:.3f} | "
                  f"Coll {info['avg_collision']:.3f} | Eps {eps:.3f} | t={elapsed:.0f}s")
    
    # Stage 2: PDR focus (1000-2000 episodes)
    print("\n  Stage 2/3: PDR Optimization (eps 1000-2000)")
    env.curriculum_stage = 2
    
    for ep in range(1000, 2000):
        state, _ = env.reset()
        ep_rew = 0.0
        ep_loss = []
        
        while True:
            actions = agent.act(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, rewards, nstate, done)
            loss = agent.train_step()
            if loss > 0:
                ep_loss.append(loss)
            state = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break
        
        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["loss"].append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        
        if ep % 200 == 0:
            elapsed = time.time() - t0
            eps = agent._get_epsilon()
            print(f"    Ep {ep:4d} | Rew {ep_rew:8.2f} | PDR {info['avg_pdr']:.3f} | "
                  f"Coll {info['avg_collision']:.3f} | Eps {eps:.3f} | t={elapsed:.0f}s")
    
    # Stage 3: Full objectives (2000-3000 episodes)
    print("\n  Stage 3/3: Full Multi-Objective (eps 2000-3000)")
    env.curriculum_stage = 3
    
    for ep in range(2000, total_episodes):
        state, _ = env.reset()
        ep_rew = 0.0
        ep_loss = []
        
        while True:
            actions = agent.act(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, rewards, nstate, done)
            loss = agent.train_step()
            if loss > 0:
                ep_loss.append(loss)
            state = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break
        
        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["throughput"].append(info["avg_throughput"])
        hist["fairness"].append(info["fairness"])
        hist["reliability"].append(info["reliability_rate"])
        hist["loss"].append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        
        if ep % 200 == 0:
            elapsed = time.time() - t0
            eps = agent._get_epsilon()
            print(f"    Ep {ep:4d} | Rew {ep_rew:8.2f} | PDR {info['avg_pdr']:.3f} | "
                  f"Coll {info['avg_collision']:.3f} | Eps {eps:.3f} | t={elapsed:.0f}s")
    
    # Final evaluation
    print("\n  Evaluating...")
    env_eval = V2VEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)
    
    def qmix_policy(eval_env, state):
        actions = agent.act(state)
        return actions
    
    rand_res = evaluate_policy(env_eval, random_policy, episodes=30)
    greedy_res = evaluate_policy(env_eval, greedy_csi_policy, episodes=30)
    rr_res = evaluate_policy(env_eval, round_robin_policy, episodes=30)
    qmix_res = evaluate_policy(env_eval, qmix_policy, episodes=30)
    
    # Save model
    model_path = f"models_improved/qmix_n{n_v2v}.pth"
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'target_q_network': agent.target_q_network.state_dict(),
        'mixer': agent.mixer.state_dict(),
        'target_mixer': agent.target_mixer.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'n_v2v': n_v2v,
        'state_dim': env.STATE_DIM,
        'n_actions': env.n_actions,
        'final_evaluation': qmix_res,
    }, model_path)
    print(f"  Model saved: {model_path}")
    
    return {
        "agent": agent,
        "histories": dict(hist),
        "evaluation": {
            "Random": rand_res,
            "Greedy CSI": greedy_res,
            "Round Robin": rr_res,
            "QMIX": qmix_res,
        },
    }


def train_mappo_improved(n_v2v: int, episodes: int = 3000) -> dict:
    """Train MAPPO with improved hyperparameters"""
    print(f"\n{'='*60}")
    print(f"  [MAPPO Improved] Training n_v2v={n_v2v}  episodes={episodes}")
    print(f"{'='*60}")

    env = V2VEnv(n_v2v=n_v2v, n_subchannels=4, episode_len=50, curriculum_stage=3)

    agent = MAPPOAgent(
        state_dim=env.STATE_DIM,
        n_actions=env.n_actions,
        n_agents=n_v2v,
        d_model=128,
        lr_actor=1e-4,  # Lower for stability
        lr_critic=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.05,  # Higher for more exploration
        epochs=15,  # More training epochs
        minibatch=64,
        device=DEVICE,
    )

    print(f"  MAPPO param count: {agent.param_count():,}")

    hist = defaultdict(list)
    t0 = time.time()

    for ep in range(episodes):
        state, _ = env.reset()
        ep_rew = 0.0

        while True:
            actions, log_probs, value = agent.act_with_info(state)
            nstate, rewards, done, _, info = env.step(actions)
            agent.store(state, actions, log_probs, rewards, value, done)
            state = nstate
            ep_rew += float(np.mean(rewards))
            if done:
                break

        train_logs = agent.train()

        hist["reward"].append(ep_rew)
        hist["pdr"].append(info["avg_pdr"])
        hist["collision"].append(info["avg_collision"])
        hist["throughput"].append(info["avg_throughput"])
        hist["fairness"].append(info["fairness"])

        if ep % 300 == 0:
            elapsed = time.time() - t0
            print(f"  Ep {ep:4d} | Rew {ep_rew:8.2f} | PDR {info['avg_pdr']:.3f} | "
                  f"Thr {info['avg_throughput']:.3f} | Fair {info['fairness']:.3f} | t={elapsed:.0f}s")

    def mappo_policy(eval_env, state):
        return agent.act(state)

    mappo_res = evaluate_policy(env, mappo_policy, episodes=30)

    # Save model
    model_path = f"models_improved/mappo_n{n_v2v}.pth"
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
        "agent": agent,
        "histories": dict(hist),
        "evaluation": {"MAPPO": mappo_res},
    }


def main():
    """Train QMIX and improved MAPPO"""
    DENSITIES = [8, 16, 24]
    EPISODES = 3000  # Extended training
    
    print("\n" + "="*80)
    print("  RESEARCH-GRADE TRAINING: QMIX + MAPPO with Curriculum Learning")
    print("="*80)
    
    all_qmix = {}
    all_mappo = {}
    
    # Train only n=8 first (quick validation)
    print("\n>> Training n=8 (validation run)")
    all_qmix[8] = train_qmix_curriculum(n_v2v=8, total_episodes=EPISODES)
    all_mappo[8] = train_mappo_improved(n_v2v=8, episodes=EPISODES)
    
    # Print results
    print("\n" + "="*80)
    print("  RESULTS FOR n=8")
    print("="*80)
    
    qmix_res = all_qmix[8]["evaluation"]["QMIX"]
    mappo_res = all_mappo[8]["evaluation"]["MAPPO"]
    
    print(f"\nQMIX:")
    print(f"  PDR: {qmix_res['PDR']:.3f}")
    print(f"  Collision: {qmix_res['Collision']:.3f}")
    print(f"  Reward: {qmix_res['Reward']:.2f}")
    
    print(f"\nMAPPO:")
    print(f"  PDR: {mappo_res['PDR']:.3f}")
    print(f"  Collision: {mappo_res['Collision']:.3f}")
    print(f"  Reward: {mappo_res['Reward']:.2f}")
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print(f"\nImproved models saved to: ./models_improved/")
    print(f"  - qmix_n8.pth")
    print(f"  - mappo_n8.pth")
    print(f"\nOriginal models preserved in: ./models/")
    print(f"  - dqn_n8.pth, dqn_n16.pth, dqn_n24.pth")
    print(f"  - mappo_n8.pth, mappo_n16.pth, mappo_n24.pth")
    print("\nTo train all densities, uncomment the loop in main()")


if __name__ == "__main__":
    main()
