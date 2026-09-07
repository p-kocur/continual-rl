import torch
import numpy as np
from ship_env import ShipSailingEnv
from baseline_ppo import PPOTrainer

def evaluate_ppo(agent, env, episodes=5):
    env.render_mode = "ansi"
    successes = 0
    avg_return = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_return = 0
        for _ in range(env.max_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(agent.device)
                action, _, _ = agent.policy_old.act(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            print(f"Step: {env.step_count}, State: {env.state}, Action: {action}, Reward: {reward}")
            print(env.render())

            if terminated:
                successes += 1
                break
            if truncated:
                break

        avg_return += ep_return

    return successes, avg_return / episodes

def main():
    env = ShipSailingEnv(enable_wind=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 10D state space, 4 discrete actions
    agent = PPOTrainer(
        state_dim=10,
        action_dim=4,
        lr_actor=0.0005,
        lr_critic=0.002,
        gamma=0.99,
        K_epochs=20,
        eps_clip=0.2,
        device=device
    )

    epochs = 70
    steps_per_epoch = 1000
    update_timestep = 1000
    time_step = 0

    print("--- Training PPO ---")
    for epoch in range(epochs):
        epoch_reward = 0
        obs, _ = env.reset()

        for t in range(steps_per_epoch):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            # Scale reward for PPO stability
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
    succ, ret = evaluate_ppo(agent, env, episodes=10)
    print(f"PPO | Success Rate: {succ}/10 | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
