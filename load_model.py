"""
Helper script to load trained models and run inference

Usage:
    python load_model.py --agent dqn --n_v2v 8
    python load_model.py --agent mappo --n_v2v 16
"""

import argparse
import torch
import numpy as np
from env import V2VEnv
from networks import TransformerDQN, MAPPOActor, MAPPOCritic


def load_dqn_model(model_path: str, device: str = "cuda"):
    """Load a trained DQN model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct network architecture
    model = TransformerDQN(
        state_dim=checkpoint['state_dim'],
        n_actions=checkpoint['n_actions'],
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        noisy=False,
    ).to(device)
    
    model.load_state_dict(checkpoint['online_net'])
    model.eval()
    
    print(f"✅ Loaded DQN model (n_v2v={checkpoint['n_v2v']})")
    print(f"   Final PDR: {checkpoint['final_evaluation']['PDR']:.3f}")
    print(f"   Final Reward: {checkpoint['final_evaluation']['Reward']:.2f}")
    
    return model, checkpoint


def load_mappo_model(model_path: str, n_agents: int, device: str = "cuda"):
    """Load a trained MAPPO model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct actor and critic
    actor = MAPPOActor(
        state_dim=checkpoint['state_dim'],
        n_actions=checkpoint['n_actions'],
        d_model=128,
    ).to(device)
    
    critic = MAPPOCritic(
        state_dim=checkpoint['state_dim'],
        n_agents=n_agents,
        d_model=256,
    ).to(device)
    
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    actor.eval()
    critic.eval()
    
    print(f"✅ Loaded MAPPO model (n_v2v={checkpoint['n_v2v']})")
    # Handle both formats: nested under 'MAPPO' or direct
    eval_data = checkpoint['final_evaluation']
    if 'MAPPO' in eval_data:
        eval_data = eval_data['MAPPO']
    print(f"   Final PDR: {eval_data['PDR']:.3f}")
    print(f"   Final Reward: {eval_data['Reward']:.2f}")
    
    return actor, critic, checkpoint


def run_dqn_episode(env, model, device):
    """Run one episode with trained DQN model"""
    state, _ = env.reset()
    total_reward = 0
    step = 0
    
    while True:
        # Greedy action selection
        actions = np.zeros(env.n_v2v, dtype=np.int64)
        model.eval()
        with torch.no_grad():
            for i in range(env.n_v2v):
                s = torch.FloatTensor(state[i:i+1]).unsqueeze(0).to(device)
                q = model(s)
                actions[i] = q.squeeze().argmax().item()
        
        state, rewards, done, _, info = env.step(actions)
        total_reward += float(np.mean(rewards))
        step += 1
        
        if done:
            break
    
    return total_reward, info


def run_mappo_episode(env, actor, device):
    """Run one episode with trained MAPPO model"""
    state, _ = env.reset()
    total_reward = 0
    step = 0
    
    while True:
        # Get actions from actor
        actions = np.zeros(env.n_v2v, dtype=np.int64)
        actor.eval()
        with torch.no_grad():
            for i in range(env.n_v2v):
                s = torch.FloatTensor(state[i:i+1]).to(device)
                dist = actor.get_dist(s)
                actions[i] = dist.sample().item()
        
        state, rewards, done, _, info = env.step(actions)
        total_reward += float(np.mean(rewards))
        step += 1
        
        if done:
            break
    
    return total_reward, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=["dqn", "mappo"])
    parser.add_argument("--n_v2v", type=int, required=True, choices=[8, 16, 24])
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model_path = f"models/{args.agent}_n{args.n_v2v}.pth"
    
    if args.agent == "dqn":
        model, checkpoint = load_dqn_model(model_path, device)
    else:
        actor, critic, checkpoint = load_mappo_model(model_path, args.n_v2v, device)
    
    # Create environment
    env = V2VEnv(n_v2v=args.n_v2v, n_subchannels=4, episode_len=50)
    
    # Run test episodes
    print(f"\n🚀 Running {args.episodes} test episodes...")
    rewards = []
    pdrs = []
    collisions = []
    
    for ep in range(args.episodes):
        if args.agent == "dqn":
            reward, info = run_dqn_episode(env, model, device)
        else:
            reward, info = run_mappo_episode(env, actor, device)
        
        rewards.append(reward)
        pdrs.append(info['avg_pdr'])
        collisions.append(info['avg_collision'])
        
        print(f"  Ep {ep+1:2d} | Reward: {reward:7.2f} | PDR: {info['avg_pdr']:.3f} | Collision: {info['avg_collision']:.3f}")
    
    # Summary
    print(f"\n📊 Summary over {args.episodes} episodes:")
    print(f"   Avg Reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Avg PDR:       {np.mean(pdrs):.3f} ± {np.std(pdrs):.3f}")
    print(f"   Avg Collision: {np.mean(collisions):.3f} ± {np.std(collisions):.3f}")


if __name__ == "__main__":
    main()
