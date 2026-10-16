import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from quadrotor_env import QuadrotorEnv
from quadrotor_prior import QuadrotorEnvVNDWrapper, f_prior_quadrotor, reward_fn_quadrotor
from continual_vnd import ContinualVNDTrainer
from baseline_ppo import ContinuousPPOTrainer
from baseline_mpc import MPCTrainer
from baseline_grad_mpc import GradMPCTrainer

# State dim: [pos(3), vel(3), ref×15(3 each)] = 51D
# Action dim: [dx, dy, dz] = 3D  (position offset from reference, bounded ±0.15 m)
STATE_DIM = 51
ACTION_DIM = 3
ACTION_BOUND = 0.15  # m, matches QuadrotorEnv action_space



def get_vnd_context(history, context_len):
    if len(history) < context_len:
        pad = context_len - len(history)
        states = [np.zeros(STATE_DIM)] * pad + [h[0] for h in history]
        actions = [np.zeros(ACTION_DIM)] * pad + [h[1] for h in history]
    else:
        states = [h[0] for h in history[-context_len:]]
        actions = [h[1] for h in history[-context_len:]]

    ctx_states = torch.FloatTensor(np.array(states)).unsqueeze(0)
    ctx_actions = torch.FloatTensor(np.array(actions)).unsqueeze(0)
    return ctx_states, ctx_actions


def evaluate_agent(agent_name, agent, env, max_steps, device="cpu", is_vnd=False, seed=42):
    obs, _ = env.reset(seed=seed)
    ep_return = 0
    history = []

    xs, ys, zs = [], [], []

    if hasattr(agent, "action_seq"):
        agent.action_seq.zero_()

    z_ema = None
    z_alpha = 0.15

    for step in range(max_steps):
        true_state = obs

        xs.append(true_state[0])
        ys.append(true_state[1])
        zs.append(true_state[2])

        if agent_name == "Continual VND":
            ctx_states, ctx_actions = get_vnd_context(history, agent.context_len)
            with torch.no_grad():
                z_raw = agent.encoder(ctx_states.to(device), ctx_actions.to(device))
                if z_ema is None:
                    z_ema = z_raw
                else:
                    z_ema = (1.0 - z_alpha) * z_ema + z_alpha * z_raw
                a = agent.policy(torch.FloatTensor(true_state).unsqueeze(0).to(device), z_ema)
                # Policy outputs Tanh → [-1,1]; scale to action bounds
                action = (a * ACTION_BOUND).squeeze(0).cpu().numpy()
        elif agent_name == "PPO":
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_mean = agent.policy_old.actor(obs_tensor)
                # Actor ends with Tanh → [-1,1]; rescale to action bounds
                action = (action_mean * ACTION_BOUND).squeeze(0).cpu().numpy()
        else:
            # MPC agents: select_action returns values in [-action_bound, action_bound]
            action = agent.select_action(true_state)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward

        if agent_name == "Continual VND":
            history.append((true_state, action))

        obs = next_obs

        if terminated or truncated:
            print(f"  [{agent_name}] episode ended at step {step+1}: term={terminated} trunc={truncated}")
            break

    return ep_return, xs, ys, zs


def plot_results(results, env):
    os.makedirs("plots", exist_ok=True)

    # Precompute reference trajectory
    ref_xs = [env._get_reference(t)[0] for t in range(env.max_steps)]
    ref_ys = [env._get_reference(t)[1] for t in range(env.max_steps)]

    colors = ['r', 'b', 'g', 'm']

    # XY trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(ref_xs, ref_ys, 'k--', label='Reference', linewidth=2)
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(data['xs'], data['ys'], label=name, color=colors[idx], alpha=0.8, linewidth=1.5)
    plt.title("PyBullet Quadrotor XY Trajectory Tracking (No Wind)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/comparison_trajectories_pybullet.png", dpi=300)
    plt.close()

    # Bar chart of scores
    names = list(results.keys())
    scores = [data['score'] for data in results.values()]
    plt.figure(figsize=(8, 6))
    plt.bar(names, scores, color=colors)
    plt.title("Average Tracking Return (PyBullet, No Wind)")
    plt.ylabel("Return (Higher is better)")
    for i, v in enumerate(scores):
        offset = -5 if v < 0 else 1
        plt.text(i, v + offset, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.savefig("plots/comparison_scores_pybullet.png", dpi=300)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # WIND DISABLED
    env = QuadrotorEnv(enable_wind=False)
    vnd_env = QuadrotorEnvVNDWrapper(env)

    total_steps = 25000
    steps_per_epoch = 1000
    epochs = total_steps // steps_per_epoch

    # ------------------------------------------------------------------
    print("\n--- Training PPO ---")
    ppo = ContinuousPPOTrainer(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        lr_actor=0.0003, lr_critic=0.001,
        gamma=0.99, K_epochs=40, eps_clip=0.2, device=device
    )
    time_step = 0
    for ep in range(epochs):
        obs, _ = env.reset()
        for t in range(steps_per_epoch):
            # PPO select_action outputs ±5; divide to fit ±ACTION_BOUND
            action_raw = ppo.select_action(obs)
            action = action_raw / 5.0 * ACTION_BOUND
            obs, reward, terminated, truncated, _ = env.step(action)
            ppo.buffer.rewards.append(reward * 0.1)
            ppo.buffer.is_terminals.append(terminated or truncated)
            time_step += 1
            if time_step % steps_per_epoch == 0:
                ppo.update()
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(f"  [PPO] Epoch {ep+1}/{epochs}")
            if terminated or truncated:
                obs, _ = env.reset()

    # ------------------------------------------------------------------
    print("\n--- Training Continual VND ---")
    vnd = ContinualVNDTrainer(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        f_prior=f_prior_quadrotor, reward_fn=reward_fn_quadrotor,
        context_len=10, latent_dim=16, hidden_dim=64,
        action_scale=ACTION_BOUND, dyn_dim=6, z_ema_alpha=0.15, device=device
    )
    for _ in range(3):
        vnd.collect_experience(vnd_env, num_steps=steps_per_epoch)

    for ep in range(epochs):
        for _ in range(100):
            vnd.update_vnd(batch_size=128)
        loss_p = 0.0
        for _ in range(100):
            res = vnd.update_policy(rollout_length=15, batch_size=128)
            loss_p = res["loss_policy"]
        vnd.collect_experience(vnd_env, num_steps=steps_per_epoch)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [VND] Epoch {ep+1}/{epochs} | Policy Loss: {loss_p:.4f}")

    # ------------------------------------------------------------------
    print("\n--- Initializing MPC Baselines ---")
    cem_mpc = MPCTrainer(
        action_dim=ACTION_DIM, f_prior=f_prior_quadrotor,
        reward_fn=reward_fn_quadrotor, horizon=15, num_particles=200,
        num_iters=3, device=device
    )
    cem_mpc.action_min = -ACTION_BOUND
    cem_mpc.action_max = ACTION_BOUND

    grad_mpc = GradMPCTrainer(
        action_dim=ACTION_DIM, f_prior=f_prior_quadrotor,
        reward_fn=reward_fn_quadrotor, horizon=15, num_iters=25,
        lr=0.1, device=device
    )
    grad_mpc.action_min = -ACTION_BOUND
    grad_mpc.action_max = ACTION_BOUND

    # ------------------------------------------------------------------
    print("\n--- Evaluating All Algorithms (No Wind) ---")
    results = {}

    agents = [
        ("PPO",          ppo,      False),
        ("CEM MPC",      cem_mpc,  True),
        ("Grad MPC",     grad_mpc, True),
        ("Continual VND", vnd,     True),
    ]

    for name, agent, is_vnd in agents:
        print(f"Evaluating {name}...")
        eval_env = vnd_env if is_vnd else env
        ret, xs, ys, zs = evaluate_agent(
            name, agent, eval_env, env.max_steps, device, is_vnd, seed=42
        )
        results[name] = {'score': ret, 'xs': xs, 'ys': ys, 'zs': zs}
        print(f"  {name} Return: {ret:.2f}  (steps: {len(xs)})")

    print("\nGenerating Plots...")
    plot_results(results, env)
    print("Done. Saved to plots/comparison_trajectories_pybullet.png and plots/comparison_scores_pybullet.png")


if __name__ == "__main__":
    main()
