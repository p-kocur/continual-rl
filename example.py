import torch
import numpy as np
import os
from continual_vnd import ContinualVNDTrainer
from ship_env import ShipSailingEnv
from compare import f_prior, reward_fn, TrueStateEnvWrapper

def main():
    state_dim = 2
    action_dim = 2

    env = ShipSailingEnv()
    vnd_env = TrueStateEnvWrapper(env)

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

    print("Collecting initial experience...")
    for _ in range(5):
        trainer.collect_experience(vnd_env, num_steps=200)

    epochs = 5

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        vnd_losses = []
        for _ in range(50):
            logs = trainer.update_vnd(batch_size=64)
            if logs:
                vnd_losses.append(logs['loss_vnd_total'])

        if vnd_losses:
            print(f"Average VND Loss: {np.mean(vnd_losses):.4f}")

        policy_losses = []
        for _ in range(50):
            logs = trainer.update_policy(rollout_length=15, batch_size=64)
            if logs:
                policy_losses.append(logs['loss_policy'])

        if policy_losses:
            print(f"Average Policy Loss: {np.mean(policy_losses):.4f}")

        avg_reward = trainer.collect_experience(vnd_env, num_steps=200)
        print(f"Collected Experience Avg Reward: {avg_reward:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(trainer.encoder.state_dict(), "checkpoints/encoder.pth")
    torch.save(trainer.policy.state_dict(), "checkpoints/policy.pth")
    print("\nModels saved to checkpoints directory.")

if __name__ == "__main__":
    main()
