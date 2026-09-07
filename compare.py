import torch
import numpy as np
from ship_env import ShipSailingEnv
from continual_vnd import ContinualVNDTrainer
from baseline_ppo import PPOTrainer

# For Continual VND, the f_prior needs to simulate the true physics (2D coordinates)
# We will use the true_state (2D) for the trainer instead of the 10D raycast observation
# because the analytical physics prior is s_{t+1} = s_t + a_t
def f_prior(state, action):
    return state + action

def reward_fn(state, action, next_state):
    target = torch.tensor([18.0, 18.0], dtype=torch.float32, device=state.device)
    dist = torch.norm(next_state - target, dim=-1)
    bounds_penalty = torch.sum(torch.relu(-next_state) + torch.relu(next_state - 20.0), dim=-1)
    return -dist * 0.1 - bounds_penalty * 2.0

class TrueStateEnvWrapper:
    # A wrapper for VND that exposes the 2D state instead of the 10D raycasts
    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps
    def reset(self):
        obs, info = self.env.reset()
        return info["true_state"], info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return info["true_state"], reward, terminated, truncated, info

def train_vnd(env, device, epochs=10, steps_per_epoch=200):
    print("\n--- Training Continual VND ---")
    vnd_env = TrueStateEnvWrapper(env)
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

    for _ in range(5):
        trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

    for epoch in range(epochs):
        for _ in range(50): trainer.update_vnd(batch_size=64)
        for _ in range(50): trainer.update_policy(rollout_length=15, batch_size=64)
        avg_reward = trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)
        print(f"Epoch {epoch+1}, Avg Reward: {avg_reward:.2f}")
    return trainer

def train_ppo(env, device, epochs=10, steps_per_epoch=200):
    print("\n--- Training PPO ---")
    ppo_agent = PPOTrainer(
        state_dim=10, # Uses the 10D raycast observation
        action_dim=4,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=40,
        eps_clip=0.2,
        device=device
    )

    update_timestep = steps_per_epoch
    time_step = 0

    for epoch in range(epochs):
        epoch_reward = 0
        state, _ = env.reset()

        for t in range(steps_per_epoch):
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward * 0.1)
            ppo_agent.buffer.is_terminals.append(terminated or truncated)
            epoch_reward += reward
            time_step += 1

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if terminated or truncated:
                state, _ = env.reset()
        print(f"Epoch {epoch+1}, Cumulative Reward: {epoch_reward:.2f}")
    return ppo_agent

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

def evaluate_ppo(agent, env, episodes=5):
    successes, avg_return = 0, 0
    for _ in range(episodes):
        state, _ = env.reset()
        ep_return = 0
        for _ in range(env.max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(agent.device)
                action, _, _ = agent.policy_old.act(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            if terminated:
                successes += 1
                break
            if truncated:
                break
        avg_return += ep_return
    return successes, avg_return / episodes

def main():
    env = ShipSailingEnv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vnd_trainer = train_vnd(env, device, epochs=5, steps_per_epoch=200)
    ppo_agent = train_ppo(env, device, epochs=5, steps_per_epoch=200)

    print("\n--- Evaluation ---")
    vnd_succ, vnd_ret = evaluate_vnd(vnd_trainer, env, episodes=10)
    ppo_succ, ppo_ret = evaluate_ppo(ppo_agent, env, episodes=10)

    print(f"Continual VND | Success Rate: {vnd_succ}/10 | Avg Return: {vnd_ret:.2f}")
    print(f"PPO           | Success Rate: {ppo_succ}/10 | Avg Return: {ppo_ret:.2f}")

if __name__ == "__main__":
    main()
