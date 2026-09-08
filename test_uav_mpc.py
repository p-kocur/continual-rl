import torch
import numpy as np
from uav_env import UAVEnv
from test_uav_vnd import f_prior_uav, reward_fn_uav
from baseline_mpc import MPCTrainer
from visualize import plot_uav_trajectory

def evaluate_mpc(agent, env, episodes=5):
    avg_return = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0

        uav_xs, uav_ys = [], []
        ref_xs, ref_ys = [], []

        for step in range(env.max_steps):
            uav_xs.append(obs[0])
            uav_ys.append(obs[1])
            ref_xs.append(obs[4])
            ref_ys.append(obs[5])

            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        if ep == 0:
            plot_uav_trajectory(uav_xs, uav_ys, ref_xs, ref_ys,
                                "MPC CEM - UAV Trajectory Tracking (No Wind)",
                                "trajectory_mpc.png")

        print(f"Episode {ep+1}/{episodes} Return: {ep_return:.2f}")
        avg_return += ep_return

    return avg_return / episodes

def main():
    env = UAVEnv(enable_wind=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = MPCTrainer(
        action_dim=2,
        f_prior=f_prior_uav,
        reward_fn=reward_fn_uav,
        horizon=15,
        num_particles=200,
        num_iters=3,
        device=device
    )

    print("\n--- Evaluating MPC on UAV ---")
    # MPC doesn't need to be trained! It uses the perfect analytical prior online.
    ret = evaluate_mpc(agent, env, episodes=5)
    print(f"\nMPC | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
