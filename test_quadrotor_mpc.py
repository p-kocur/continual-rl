import torch
import numpy as np
from quadrotor_env import QuadrotorEnv
from quadrotor_prior import QuadrotorEnvVNDWrapper, f_prior_quadrotor, reward_fn_quadrotor
from baseline_grad_mpc import GradMPCTrainer
from visualize import plot_uav_trajectory

def evaluate_quadrotor_mpc(agent, vnd_env, episodes=1):
    avg_return = 0

    for ep in range(episodes):
        obs, _ = vnd_env.reset()
        ep_return = 0

        # Reset warm-start
        agent.action_seq.zero_()

        uav_xs, uav_ys = [], []
        ref_xs, ref_ys = [], []

        for step in range(vnd_env.max_steps):
            # Extract x, y for plotting
            uav_xs.append(obs[0])
            uav_ys.append(obs[1])
            ref_xs.append(obs[12])
            ref_ys.append(obs[13])

            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = vnd_env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        if ep == 0:
            # We plot the X-Y projection of the 3D flight
            plot_uav_trajectory(uav_xs, uav_ys, ref_xs, ref_ys,
                                "Grad-MPC - PyBullet Quadrotor (No Wind)",
                                "trajectory_pybullet_mpc.png")

        print(f"Episode {ep+1}/{episodes} Return: {ep_return:.2f}")
        avg_return += ep_return

    return avg_return / episodes

def main():
    env = QuadrotorEnv(enable_wind=False)
    vnd_env = QuadrotorEnvVNDWrapper(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = GradMPCTrainer(
        action_dim=4, # 4 motors
        f_prior=f_prior_quadrotor,
        reward_fn=reward_fn_quadrotor,
        horizon=10,
        num_iters=25,
        lr=0.1,
        device=device
    )

    # Bound the policy action correctly (policy assumes [-1, 1] range which env maps to RPM)
    agent.action_min = -1.0
    agent.action_max = 1.0

    print("\n--- Evaluating Gradient MPC on PyBullet Quadrotor ---")
    ret = evaluate_quadrotor_mpc(agent, vnd_env, episodes=2)
    print(f"\nGradient MPC | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
