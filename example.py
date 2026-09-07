import torch
import numpy as np
import os
from continual_vnd import ContinualVNDTrainer
from ship_env import ShipSailingEnv

# PyTorch differentiable nominal physics prior: s_{t+1} = s_t + a_t
def f_prior(state, action):
    return state + action

# Differentiable reward function to train the policy over simulated rollouts
def reward_fn(state, action, next_state):
    # Differentiable version of the environment reward
    # state/next_state shape: (B, 2)
    target = torch.tensor([45.0, 45.0], dtype=torch.float32, device=state.device)

    # Distance penalty
    dist = torch.norm(next_state - target, dim=-1)

    # Simple bounds penalty to keep it in the 50x50 grid
    bounds_penalty = torch.sum(torch.relu(-next_state) + torch.relu(next_state - 50.0), dim=-1)

    # We omit hard obstacle collision penalties here to keep the reward function smooth
    # The latent dynamics D(s,a,z) should ideally predict collisions and stop the ship.
    return -dist * 0.1 - bounds_penalty * 2.0

def main():
    state_dim = 2
    action_dim = 2

    env = ShipSailingEnv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = ContinualVNDTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        f_prior=f_prior,
        reward_fn=reward_fn,
        context_len=5,
        latent_dim=16,
        hidden_dim=64,
        buffer_capacity=10000,
        device=device
    )

    # 1. Collect some initial experience (random or poorly initialized policy)
    print("Collecting initial experience...")
    for _ in range(5):
        trainer.collect_experience(env, num_steps=200)

    epochs = 10

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # 2. Train VND model
        vnd_losses = []
        for _ in range(50):
            logs = trainer.update_vnd(batch_size=64)
            if logs:
                vnd_losses.append(logs['loss_vnd_total'])

        if vnd_losses:
            print(f"Average VND Loss: {np.mean(vnd_losses):.4f}")

        # 3. Train Policy
        policy_losses = []
        for _ in range(50):
            logs = trainer.update_policy(rollout_length=15, batch_size=64)
            if logs:
                policy_losses.append(logs['loss_policy'])

        if policy_losses:
            print(f"Average Policy Loss: {np.mean(policy_losses):.4f}")

        # 4. Collect more experience with updated policy
        avg_reward = trainer.collect_experience(env, num_steps=200)
        print(f"Collected Experience Avg Reward: {avg_reward:.2f}")

    # Save models
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(trainer.encoder.state_dict(), "checkpoints/encoder.pth")
    torch.save(trainer.policy.state_dict(), "checkpoints/policy.pth")
    print("\nModels saved to checkpoints directory.")

if __name__ == "__main__":
    main()
