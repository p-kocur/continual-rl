import torch
import numpy as np

# A dummy gym-like environment interface for testing
class DummyEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = np.zeros(state_dim)

    def reset(self):
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # A simple linear transition: s_{t+1} = s_t + a_t + noise
        self.state = self.state + np.pad(action, (0, self.state_dim - self.action_dim)) + np.random.randn(self.state_dim).astype(np.float32) * 0.1
        reward = -np.sum(self.state**2)  # Try to keep state near origin
        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, {}

# User-provided analytic prior: assume no change, s_{t+1} = s_t
def f_prior(state, action):
    return state

# User-provided differentiable reward function for BPTT
def reward_fn(state, action, next_state):
    # Differentiable version of the environment reward
    return -torch.sum(next_state**2, dim=-1)

def main():
    from continual_vnd import ContinualVNDTrainer

    state_dim = 4
    action_dim = 2
    env = DummyEnv(state_dim, action_dim)

    trainer = ContinualVNDTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        f_prior=f_prior,
        reward_fn=reward_fn,
        context_len=5,
        latent_dim=8,
        hidden_dim=32,
        buffer_capacity=1000
    )

    # 1. Collect some initial experience
    print("Collecting initial experience...")
    trainer.collect_experience(env, num_steps=200)

    # 2. Train VND
    print("Training VND model...")
    for _ in range(50):
        vnd_logs = trainer.update_vnd(batch_size=32)
    print("VND logs:", vnd_logs)

    # 3. Train Policy using differentiable simulator
    print("Training Policy...")
    for _ in range(50):
        policy_logs = trainer.update_policy(rollout_length=10, batch_size=32)
    print("Policy logs:", policy_logs)

if __name__ == "__main__":
    main()
