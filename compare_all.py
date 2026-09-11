import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from uav_env import UAVEnv
from test_uav_vnd import f_prior_uav, reward_fn_uav, UAVEnvVNDWrapper
from continual_vnd import ContinualVNDTrainer
from baseline_ppo import ContinuousPPOTrainer
from baseline_mpc import MPCTrainer
from baseline_grad_mpc import GradMPCTrainer

def get_vnd_context(history, context_len):
    if len(history) < context_len:
        pad = context_len - len(history)
        states = [np.zeros(24)] * pad + [h[0] for h in history]
        actions = [np.zeros(2)] * pad + [h[1] for h in history]
    else:
        states = [h[0] for h in history[-context_len:]]
        actions = [h[1] for h in history[-context_len:]]

    ctx_states = torch.FloatTensor(np.array(states)).unsqueeze(0)
    ctx_actions = torch.FloatTensor(np.array(actions)).unsqueeze(0)
    return ctx_states, ctx_actions

def evaluate_agent(agent_name, agent, env, max_steps, device="cpu", is_vnd=False):
    # env was seeded prior to this call, do not overwrite the seed
    obs, info = env.reset()
    ep_return = 0
    history = []

    uav_xs, uav_ys = [], []
    ref_xs, ref_ys = [], []

    # Reset warm-start for GradMPC if applicable
    if hasattr(agent, "action_seq"):
        agent.action_seq.zero_()

    for step in range(max_steps):
        # For VND, true_state must be the 24D packed vector (obs)
        true_state = obs if is_vnd else obs

        uav_xs.append(true_state[0])
        uav_ys.append(true_state[1])
        ref_xs.append(true_state[4])
        ref_ys.append(true_state[5])

        if agent_name == "Continual VND":
            ctx_states, ctx_actions = get_vnd_context(history, agent.context_len)
            with torch.no_grad():
                z = agent.encoder(ctx_states.to(device), ctx_actions.to(device))
                a = agent.policy(torch.FloatTensor(true_state).unsqueeze(0).to(device), z)
                action = (a * 5.0).squeeze(0).cpu().numpy()
        elif agent_name == "PPO":
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_mean = agent.policy_old.actor(obs_tensor)
                action = (action_mean * 5.0).squeeze(0).cpu().numpy()
        else:
            # MPC or Grad-MPC
            action = agent.select_action(true_state)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        ep_return += reward

        if agent_name == "Continual VND":
            history.append((true_state, action))

        obs = next_obs
        info = next_info

        if terminated or truncated:
            break

    return ep_return, uav_xs, uav_ys, ref_xs, ref_ys

def plot_results(results, ref_xs, ref_ys):
    os.makedirs("plots", exist_ok=True)

    # 1. Plot Trajectories
    plt.figure(figsize=(10, 8))
    plt.plot(ref_xs, ref_ys, 'k--', label='Reference', linewidth=2)

    colors = ['r', 'b', 'g', 'm']
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(data['xs'], data['ys'], label=name, color=colors[idx], alpha=0.8, linewidth=1.5)

    plt.title("UAV Trajectory Tracking (Wind Enabled)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("plots/comparison_trajectories.png", dpi=300)
    plt.close()

    # 2. Plot Bar Chart of Scores
    names = list(results.keys())
    scores = [data['score'] for name, data in results.items()]

    plt.figure(figsize=(8, 6))
    plt.bar(names, scores, color=colors)
    plt.title("Average Tracking Return (Wind Enabled)")
    plt.ylabel("Return (Higher is better)")

    # Annotate bars
    for i, v in enumerate(scores):
        plt.text(i, v - 50, f"{v:.1f}", ha='center', va='bottom', color='white', fontweight='bold')

    plt.savefig("plots/comparison_scores.png", dpi=300)
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # WIND IS ENABLED
    env = UAVEnv(enable_wind=True)
    vnd_env = UAVEnvVNDWrapper(env)

    total_steps = 10000
    steps_per_epoch = 1000
    epochs = total_steps // steps_per_epoch

    print("\n--- Training PPO ---")
    ppo = ContinuousPPOTrainer(state_dim=24, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=40, eps_clip=0.2, device=device)
    time_step = 0
    for ep in range(epochs):
        obs, _ = env.reset()
        for t in range(steps_per_epoch):
            action = ppo.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ppo.buffer.rewards.append(reward * 0.1)
            ppo.buffer.is_terminals.append(terminated or truncated)
            time_step += 1
            if time_step % steps_per_epoch == 0: ppo.update()
            if terminated or truncated: obs, _ = env.reset()

    print("\n--- Training Continual VND ---")
    vnd = ContinualVNDTrainer(state_dim=24, action_dim=2, f_prior=f_prior_uav, reward_fn=reward_fn_uav, context_len=10, latent_dim=16, hidden_dim=64, device=device)
    # Pre-fill buffer
    for _ in range(3): vnd.collect_experience(vnd_env, num_steps=steps_per_epoch)

    for ep in range(epochs):
        for _ in range(100): vnd.update_vnd(batch_size=128)
        for _ in range(100): vnd.update_policy(rollout_length=10, batch_size=128)
        vnd.collect_experience(vnd_env, num_steps=steps_per_epoch)

    print("\n--- Initializing MPC Baselines ---")
    cem_mpc = MPCTrainer(action_dim=2, f_prior=f_prior_uav, reward_fn=reward_fn_uav, horizon=15, num_particles=200, num_iters=3, device=device)
    grad_mpc = GradMPCTrainer(action_dim=2, f_prior=f_prior_uav, reward_fn=reward_fn_uav, horizon=15, num_iters=25, lr=0.1, device=device)

    print("\n--- Evaluating All Algorithms (with Wind) ---")
    results = {}

    # We run one deterministic evaluation episode to plot exactly how they track
    # Setting seed for consistency in wind variations across the test
    # Note: Grad MPC uses true_state (24D), PPO uses obs (24D), CEM uses true_state (24D)
    # We can just say "True" for Grad MPC and CEM MPC to give them true_state (which is obs in VND wrapper, basically the 24D tensor)
    # Actually, the wrapper pack() function just returns the 24D obs. So obs == true_state.
    agents = [
        ("PPO", ppo, False),
        ("CEM MPC", cem_mpc, True), # use 24D true state for planning
        ("Grad MPC", grad_mpc, True), # use 24D true state for planning
        ("Continual VND", vnd, True)
    ]

    ref_xs_plot, ref_ys_plot = None, None

    for name, agent, is_vnd in agents:
        print(f"Evaluating {name}...")
        env.reset(seed=42) # Force same wind pattern for fair visualization
        ret, uav_xs, uav_ys, ref_xs, ref_ys = evaluate_agent(name, agent, env, env.max_steps, device, is_vnd)
        results[name] = {'score': ret, 'xs': uav_xs, 'ys': uav_ys}
        ref_xs_plot, ref_ys_plot = ref_xs, ref_ys
        print(f"{name} Return: {ret:.2f}")

    print("\nGenerating Plots...")
    plot_results(results, ref_xs_plot, ref_ys_plot)
    print("Done. Saved to plots/comparison_trajectories.png and plots/comparison_scores.png")

if __name__ == "__main__":
    main()
