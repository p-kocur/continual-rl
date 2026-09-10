import torch
import numpy as np
from uav_env import UAVEnv
from test_uav_vnd import f_prior_uav, reward_fn_uav, UAVEnvVNDWrapper
from baseline_grad_mpc import GradMPCTrainer
from visualize import plot_uav_trajectory

def evaluate_grad_mpc(agent, vnd_env, episodes=5):
    avg_return = 0

    for ep in range(episodes):
        obs, _ = vnd_env.reset()
        ep_return = 0

        # Reset warm-start
        agent.action_seq.zero_()

        uav_xs, uav_ys = [], []
        ref_xs, ref_ys = [], []

        for step in range(vnd_env.max_steps):
            uav_xs.append(obs[0])
            uav_ys.append(obs[1])
            # Target is index 4,5 at the current step
            ref_xs.append(obs[4])
            ref_ys.append(obs[5])

            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = vnd_env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        if ep == 0:
            plot_uav_trajectory(uav_xs, uav_ys, ref_xs, ref_ys,
                                "Grad-MPC (Shooting) - UAV Trajectory Tracking (No Wind)",
                                "trajectory_grad_mpc.png")

        print(f"Episode {ep+1}/{episodes} Return: {ep_return:.2f}")
        avg_return += ep_return

    return avg_return / episodes

def main():
    env = UAVEnv(enable_wind=False)
    vnd_env = UAVEnvVNDWrapper(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = GradMPCTrainer(
        action_dim=2,
        f_prior=f_prior_uav,
        reward_fn=reward_fn_uav,
        horizon=15,
        num_iters=15,
        lr=0.1,
        device=device
    )

    print("\n--- Evaluating Gradient MPC on UAV ---")
    ret = evaluate_grad_mpc(agent, vnd_env, episodes=5)
    print(f"\nGradient MPC | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
