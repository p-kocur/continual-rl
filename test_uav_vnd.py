import torch
import numpy as np
import math
from uav_env import UAVEnv
from continual_vnd import ContinualVNDTrainer
from visualize import plot_uav_trajectory

def f_prior(state, action):
    # state: [B, 4] -> px, py, vx, vy
    # action: [B, 2] -> ax, ay
    # Nominal rigid body kinematics (no wind)
    dt = 0.1

    px = state[:, 0]
    py = state[:, 1]
    vx = state[:, 2]
    vy = state[:, 3]

    ax = action[:, 0]
    ay = action[:, 1]

    new_vx = vx + ax * dt
    new_vy = vy + ay * dt

    new_px = px + new_vx * dt
    new_py = py + new_vy * dt

    return torch.stack([new_px, new_py, new_vx, new_vy], dim=-1)

# We need the reward_fn to know the reference trajectory, so we make it a class
class DifferentiableReward:
    def __init__(self, dt=0.1):
        self.dt = dt
class UAVEnvVNDWrapper:
    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps

    def _pack(self, obs, info):
        # obs is already [drone_state(4), ref0(4), ..., refH(4)]
        # We will use this vector as the 'true_state' for VND.
        return obs

    def reset(self):
        obs, info = self.env.reset()
        true_state = self._pack(obs, info)
        return true_state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        true_state = self._pack(obs, info)
        return true_state, reward, terminated, truncated, info

def f_prior_uav(state, action):
    # state: [B, 24] -> 4 drone state + 5 * 4 reference states
    dt = 0.1

    px, py = state[:, 0], state[:, 1]
    vx, vy = state[:, 2], state[:, 3]
    ax, ay = action[:, 0], action[:, 1]

    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_px = px + new_vx * dt
    new_py = py + new_vy * dt

    # We shift the reference horizon: ref_t becomes ref_{t+1}, etc.
    # The last reference point is just duplicated because we don't know the exact curve beyond the horizon
    # without explicitly recalculating the Lissajous equations here (which we could, but shifting is general).
    new_refs = state[:, 8:].clone() # take ref1 ... refH
    last_ref = state[:, -4:]        # take refH

    new_state = [new_px, new_py, new_vx, new_vy]

    # Pack back together: [new_drone_state, new_refs, last_ref]
    return torch.cat([torch.stack(new_state, dim=-1), new_refs, last_ref], dim=-1)

def reward_fn_uav(state, action, next_state):
    # next_state: [B, 24]
    px, py = next_state[:, 0], next_state[:, 1]
    vx, vy = next_state[:, 2], next_state[:, 3]

    # The current reference state for this step is stored in indices 4:8
    ref_px, ref_py = next_state[:, 4], next_state[:, 5]
    ref_vx, ref_vy = next_state[:, 6], next_state[:, 7]

    pos_error = torch.sqrt((px - ref_px)**2 + (py - ref_py)**2 + 1e-6)
    vel_error = torch.sqrt((vx - ref_vx)**2 + (vy - ref_vy)**2 + 1e-6)

    return -(pos_error + 0.1 * vel_error)

def evaluate_uav(trainer, env, episodes=5):
    vnd_env = UAVEnvVNDWrapper(env)
    trainer.encoder.eval()
    trainer.policy.eval()
    avg_return = 0

    for ep in range(episodes):
        state, _ = vnd_env.reset()
        history = []
        ep_return = 0

        uav_xs, uav_ys = [], []
        ref_xs, ref_ys = [], []

        for _ in range(vnd_env.max_steps):
            # Record trajectory
            uav_xs.append(state[0])
            uav_ys.append(state[1])
            ref_xs.append(state[4])
            ref_ys.append(state[5])

            if len(history) < trainer.context_len:
                pad = trainer.context_len - len(history)
                states = [np.zeros(24)] * pad + [h[0] for h in history]
                actions = [np.zeros(2)] * pad + [h[1] for h in history]
            else:
                states = [h[0] for h in history[-trainer.context_len:]]
                actions = [h[1] for h in history[-trainer.context_len:]]

            ctx_states = torch.FloatTensor(np.array(states)).unsqueeze(0).to(trainer.device)
            ctx_actions = torch.FloatTensor(np.array(actions)).unsqueeze(0).to(trainer.device)

            with torch.no_grad():
                z = trainer.encoder(ctx_states, ctx_actions)
                a = trainer.policy(torch.FloatTensor(state).unsqueeze(0).to(trainer.device), z)
                action = (a * 5.0).squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = vnd_env.step(action)
            ep_return += reward
            history.append((state, action))
            state = next_state

            if terminated or truncated:
                break

        # Generate plot for the very first evaluation episode
        if ep == 0:
            plot_uav_trajectory(uav_xs, uav_ys, ref_xs, ref_ys,
                                "Continual VND - UAV Trajectory Tracking (With Wind)",
                                "trajectory_vnd.png")

        avg_return += ep_return
    return avg_return / episodes

def main():
    # Enable wind to test actual VND residual learning!
    env = UAVEnv(enable_wind=True)
    vnd_env = UAVEnvVNDWrapper(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = ContinualVNDTrainer(
        state_dim=24,
        action_dim=2,
        f_prior=f_prior_uav,
        reward_fn=reward_fn_uav,
        context_len=10,
        latent_dim=16,
        hidden_dim=64,
        device=device
    )

    epochs = 10
    steps_per_epoch = 1000

    print("--- Training Continual VND on UAV ---")

    for _ in range(5):
        trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

    for epoch in range(epochs):
        vnd_losses = []
        for _ in range(100):
            logs = trainer.update_vnd(batch_size=128)
            if logs: vnd_losses.append(logs['loss_dyn'])

        pol_losses = []
        for _ in range(100):
            logs = trainer.update_policy(rollout_length=10, batch_size=128)
            if logs: pol_losses.append(logs['loss_policy'])

        avg_reward = trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

        avg_dyn = np.mean(vnd_losses) if vnd_losses else 0
        avg_pol = np.mean(pol_losses) if pol_losses else 0
        print(f"Epoch {epoch+1:02d} | Dyn Loss: {avg_dyn:.4f} | Pol Loss: {avg_pol:.4f} | Avg Reward: {avg_reward:.2f}")

    print("\n--- Evaluation ---")
    ret = evaluate_uav(trainer, env, episodes=5)
    print(f"Continual VND | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
