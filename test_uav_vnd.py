import torch
import numpy as np
import math
from uav_env import UAVEnv
from continual_vnd import ContinualVNDTrainer

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
        # We need a way to track the step t in the BPTT rollout.
        # But BPTT rollouts start from random buffer states which have different 't'
        # In the paper, the policy doesn't receive 't', it receives the ref_state in observation.

    def __call__(self, obs, action, next_obs):
        # We assume the policy's observation contains the current ref_state.
        # Wait, the trainer passes `state` to f_prior and reward_fn.
        # If we pass true_state (4D), we don't have ref_state in it.
        # The trainer expects reward_fn(state, action, next_state).

        # We have a dilemma: The true_state is 4D (px,py,vx,vy), but reward depends on tracking the reference.
        # To make BPTT work smoothly, we can put the reference state INTO the true_state that we pass to VND!
        pass

# Let's write a wrapper that makes the 'true_state' for VND contain both the drone state AND the reference state.
class UAVEnvVNDWrapper:
    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps

    def _pack(self, obs, info):
        # obs is already [drone_state, ref_state] (8D)
        # We will use this 8D vector as the 'true_state' for VND.
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
    # state: [B, 8] -> [px, py, vx, vy, ref_px, ref_py, ref_vx, ref_vy]
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

    # For the reference state, the agent can't affect it.
    # But during BPTT rollout, we need the reference state to advance forward in time!
    # Because BPTT rolls out multiple steps into the future.
    # If we don't know 't', we can roughly approximate ref_next = ref_curr + ref_v * dt
    ref_px = state[:, 4]
    ref_py = state[:, 5]
    ref_vx = state[:, 6]
    ref_vy = state[:, 7]

    # Simple linear extrapolation for the reference trajectory for short BPTT horizons
    # This is a common trick when 't' is not explicitly available.
    new_ref_px = ref_px + ref_vx * dt
    new_ref_py = ref_py + ref_vy * dt
    # Assume constant reference velocity for the short horizon
    new_ref_vx = ref_vx
    new_ref_vy = ref_vy

    return torch.stack([new_px, new_py, new_vx, new_vy, new_ref_px, new_ref_py, new_ref_vx, new_ref_vy], dim=-1)

def reward_fn_uav(state, action, next_state):
    # next_state: [B, 8]
    px, py = next_state[:, 0], next_state[:, 1]
    vx, vy = next_state[:, 2], next_state[:, 3]
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

    for _ in range(episodes):
        state, _ = vnd_env.reset()
        history = []
        ep_return = 0

        for _ in range(vnd_env.max_steps):
            if len(history) < trainer.context_len:
                pad = trainer.context_len - len(history)
                states = [np.zeros(8)] * pad + [h[0] for h in history]
                actions = [np.zeros(2)] * pad + [h[1] for h in history]
            else:
                states = [h[0] for h in history[-trainer.context_len:]]
                actions = [h[1] for h in history[-trainer.context_len:]]

            ctx_states = torch.FloatTensor(np.array(states)).unsqueeze(0).to(trainer.device)
            ctx_actions = torch.FloatTensor(np.array(actions)).unsqueeze(0).to(trainer.device)

            with torch.no_grad():
                z = trainer.encoder(ctx_states, ctx_actions)
                a = trainer.policy(torch.FloatTensor(state).unsqueeze(0).to(trainer.device), z)
                action = a.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = vnd_env.step(action)
            ep_return += reward
            history.append((state, action))
            state = next_state

            if terminated or truncated:
                break
        avg_return += ep_return
    return avg_return / episodes

def main():
    # Enable wind to test actual VND residual learning!
    env = UAVEnv(enable_wind=True)
    vnd_env = UAVEnvVNDWrapper(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = ContinualVNDTrainer(
        state_dim=8,
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
