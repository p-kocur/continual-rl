import torch
import numpy as np
from uav_env import UAVEnv
from baseline_ppo import ContinuousPPOTrainer

def evaluate_ppo(agent, env, episodes=5):
    avg_return = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_return = 0

        for _ in range(env.max_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                action_mean = agent.policy_old.actor(obs_tensor)
                action = (action_mean * 5.0).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        avg_return += ep_return

    return avg_return / episodes

def main():
    env = UAVEnv(enable_wind=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Observation: 8D, Action: 2D Continuous
    agent = ContinuousPPOTrainer(
        state_dim=8,
        action_dim=2,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=40,
        eps_clip=0.2,
        device=device
    )

    epochs = 20
    steps_per_epoch = 1000
    update_timestep = 1000
    time_step = 0

    print("--- Training PPO on UAV ---")
    for epoch in range(epochs):
        epoch_reward = 0
        obs, _ = env.reset()

        for t in range(steps_per_epoch):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            # Scale reward for PPO stability (optional but helps critics)
            agent.buffer.rewards.append(reward * 0.1)
            agent.buffer.is_terminals.append(terminated or truncated)

            epoch_reward += reward
            time_step += 1

            if time_step % update_timestep == 0:
                agent.update()

            if terminated or truncated:
                obs, _ = env.reset()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}, Cumulative Reward: {epoch_reward:.2f}")

    print("\n--- Evaluation ---")
    ret = evaluate_ppo(agent, env, episodes=10)
    print(f"PPO | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
