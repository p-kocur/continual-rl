import torch
import numpy as np
from ship_env import ShipSailingEnv
from continual_vnd import ContinualVNDTrainer
from compare import TrueStateEnvWrapper

def f_prior(state, action):
    # Nominal next state
    next_state = state + action

    # Differentiable obstacles to allow BPTT to learn to route around them.
    # [x_min, x_max, y_min, y_max]
    obstacles = [
        torch.tensor([4.0, 6.0, 6.0, 12.0], device=state.device),
        torch.tensor([10.0, 12.0, 8.0, 15.0], device=state.device),
        torch.tensor([13.0, 16.0, 9.0, 12.0], device=state.device),
        torch.tensor([14.0, 20.0, 0.0, 5.0], device=state.device),
        torch.tensor([0.0, 15.0, 17.0, 20.0], device=state.device)
    ]

    repulsion = torch.zeros_like(state)
    temp = 5.0 # Sharpness of the wall

    for obs in obstacles:
        # Smooth box intersection
        in_x = torch.sigmoid(temp * (next_state[:, 0] - obs[0])) * torch.sigmoid(temp * (obs[1] - next_state[:, 0]))
        in_y = torch.sigmoid(temp * (next_state[:, 1] - obs[2])) * torch.sigmoid(temp * (obs[3] - next_state[:, 1]))

        is_inside = in_x * in_y

        # Repulsive force counters the action if entering an obstacle
        repulsion -= action * is_inside.unsqueeze(-1)

    # Also add differentiable bounds for the 20x20 grid
    bound_x_low = torch.sigmoid(temp * (-next_state[:, 0]))
    bound_x_high = torch.sigmoid(temp * (next_state[:, 0] - 20.0))
    bound_y_low = torch.sigmoid(temp * (-next_state[:, 1]))
    bound_y_high = torch.sigmoid(temp * (next_state[:, 1] - 20.0))

    out_of_bounds = torch.clamp(bound_x_low + bound_x_high + bound_y_low + bound_y_high, 0.0, 1.0)
    repulsion -= action * out_of_bounds.unsqueeze(-1)

    return next_state + repulsion

def reward_fn(state, action, next_state):
    target = torch.tensor([18.0, 20.0], dtype=torch.float32, device=state.device)
    dist = torch.norm(next_state - target, dim=-1)

    # Give a massive reward for getting close to the target to overcome the gradient plateau
    success_reward = torch.sigmoid(5.0 * (2.5 - dist)) * 50.0

    bounds_penalty = torch.sum(torch.relu(-next_state) + torch.relu(next_state - 20.0), dim=-1)
    return -dist * 0.1 - bounds_penalty * 2.0 + success_reward

def evaluate_vnd(trainer, env, episodes=5):
    vnd_env = TrueStateEnvWrapper(env)
    trainer.encoder.eval()
    trainer.policy.eval()
    successes, avg_return = 0, 0

    for _ in range(episodes):
        state, _ = vnd_env.reset()
        history = []
        ep_return = 0

        for _ in range(vnd_env.max_steps):
            if len(history) < trainer.context_len:
                pad = trainer.context_len - len(history)
                states = [np.zeros(2)] * pad + [h[0] for h in history]
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
            if terminated:
                successes += 1
                break
            if truncated:
                break
        avg_return += ep_return
    return successes, avg_return / episodes

def main():
    env = ShipSailingEnv(enable_wind=False)
    vnd_env = TrueStateEnvWrapper(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = ContinualVNDTrainer(
        state_dim=2,
        action_dim=2,
        f_prior=f_prior,
        reward_fn=reward_fn,
        context_len=5,
        latent_dim=16,
        hidden_dim=64,
        device=device
    )

    epochs = 50
    steps_per_epoch = 1000

    print("--- Training Continual VND ---")

    # Collect initial buffer
    for _ in range(5):
        trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

    for epoch in range(epochs):
        # We increase the number of optimization steps to match the 1000 steps_per_epoch intensity of PPO
        for _ in range(250): trainer.update_vnd(batch_size=64)
        for _ in range(250): trainer.update_policy(rollout_length=15, batch_size=64)

        avg_reward = trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}, Collected Experience Avg Reward: {avg_reward:.2f}")

    print("\n--- Evaluation ---")
    succ, ret = evaluate_vnd(trainer, env, episodes=10)
    print(f"Continual VND | Success Rate: {succ}/10 | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
