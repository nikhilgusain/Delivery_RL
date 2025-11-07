import os, time, random, torch
import torch.nn as nn
import torch.optim as optim
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from models.dqn import DQN, ReplayBuffer, Transition

def train_dqn(env, device, label, cfg, model_path, n_episodes, lr, eps_decay):
    """Train DQN (old/improved) on CPU or GPU and return training time."""
    start_time = time.time()
    print(f"\n🤖 Training {label} on {device} for {n_episodes} episodes")

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_obs = len(state)

    policy_net = DQN(n_obs, n_actions).to(device)
    target_net = DQN(n_obs, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayBuffer(10000)
    gamma, tau, batch_size = cfg["hyperparams"]["gamma"], cfg["hyperparams"]["tau"], cfg["hyperparams"]["batch_size"]

    eps_start, eps_end = 0.9, 0.05
    steps, rewards = 0, []

    for ep in tqdm(range(n_episodes), desc=f"{label} ({device})"):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for _ in range(env.n_deliveries + 5):
            eps = eps_end + (eps_start - eps_end) * np.exp(-steps / eps_decay)
            steps += 1
            if random.random() > eps:
                with torch.no_grad():
                    q_values = policy_net(state)
                    q_values[0, state[0, 1:].bool()] = -float("inf")
                    action = q_values.max(1)[1].view(1, 1)
            else:
                avail = [i for i, v in enumerate(env.visited_mask) if not v]
                if not avail:
                    break
                action = torch.tensor([[random.choice(avail)]], device=device, dtype=torch.long)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward_t = torch.tensor([reward], device=device)
            done = terminated or truncated
            next_state = None if done else torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward_t)
            state = next_state

            if len(memory) >= batch_size:
                batch = Transition(*zip(*memory.sample(batch_size)))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device)
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                q_vals = policy_net(state_batch).gather(1, action_batch)
                next_vals = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    nqv = target_net(non_final_next_states)
                    nqv[non_final_next_states[:, 1:].bool()] = -float("inf")
                    next_vals[non_final_mask] = nqv.max(1)[0]
                expected = (next_vals * gamma) + reward_batch

                loss = nn.SmoothL1Loss()(q_vals, expected.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

                # Soft target update
                for key, val in policy_net.state_dict().items():
                    target_net.state_dict()[key].copy_(val * tau + target_net.state_dict()[key] * (1 - tau))

            if done: break

        rewards.append(total_reward)

    dur = time.time() - start_time
    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ {label} ({device}) training done in {dur:.2f}s and saved → {model_path}")

    plt.figure(figsize=(8,4))
    plt.plot(rewards, label='Reward per Episode')
    pd.Series(rewards).rolling(10, min_periods=1).mean().plot(label='Moving Avg (10)')
    plt.title(f'{label} Rewards ({device})'); plt.legend(); plt.grid(True)
    os.makedirs(cfg["plots_dir"], exist_ok=True)
    plt.savefig(os.path.join(cfg["plots_dir"], f"{label.lower().replace(' ','_')}_{device}_rewards.png"), dpi=600)
    plt.close()
    return dur
